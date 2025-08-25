/*
 * Copyright (c) 2018 XLAB d.o.o
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*
NOTE: This file is copied from the original gofe library and edited to obtain a new labeled-key constant function hiding MIFE scheme!
*/

package schemes

import (
	"crypto/aes"
	"crypto/cipher"
	"crypto/rand"
	"fmt"
	"math/big"

	"github.com/JasZal/gofe/data"
	//"github.com/JasZal/gofe/internal/dlog"
	"github.com/JasZal/gofe/sample"
	"github.com/fentec-project/bn256"
)

// LKADOTParams represents configuration parameters for the labeled-key affine fh MIFE scheme adapted from Datta et al.
// SecLevel (int): The parameter defines the security assumption of the scheme,
// so called k-Lin assumption, where k is the specified SecLevel.
// NumClients (int): The number of clients participating
// VecLen (int): The length of vectors that clients encrypt.
// BoundX (int): The value by which the coordinates of encrypted vectors are bounded.
// BoundY (int): The value by which the coordinates of inner product vectors are bounded.
type LKADOTParams struct {
	SecLevel   int
	NumClients int
	VecLen     int
	BoundX     *big.Int
	BoundY     *big.Int
}

type LKADOTKeyStruct struct {
	TheSecKey *LKADOTSecKey
}

// LKADOT represents a Function Hiding affine Multi-input Inner Product
// Encryption scheme with labeled ciphertexts and labeled keys, modified from the paper by P. Datta, T. Okamoto, and
// J. Tomida:
// "Full-Hiding (Unbounded) Multi-Input Inner Product Functional Encryption
// from the 𝒌-Linear Assumption".
// It allows clients to encrypt vectors {x_1,...,x_m} and derive a secret key
// based on an inner product vectors {y_1,...,y_m, y_m+1} so that a decryptor can
// decrypt the sum of inner products <x_1,y_1> + ... + <x_m, y_m> + y_m+1 without
// revealing vectors x_i or y_i. The scheme is slightly modified from the
// original one to achieve a better performance. The difference is in
// storing the secret master key as matrices B, BStar, instead of matrices
// of elliptic curve elements g_1^B, g_2^BStar. This replaces elliptic curves
// operations with matrix multiplication.
//
// This struct contains the shared choice for parameters on which the
// functionality of the scheme depend.
type LKADOT struct {
	Params *LKADOTParams
}

// LKADOTSecKey represents a master secret key in LKADOT scheme.
type LKADOTSecKey struct {
	PRFkey   [][]byte
	BHat     []data.Matrix
	BStarHat []data.Matrix
}

// NewLKADOT configures a new instance of the scheme. See struct
// LKADOTParams for the description of the parameters. It returns
// a new LKADOT instance.
func NewLKADOT(secLevel, numClients, vecLen int, boundX, boundY *big.Int) *LKADOT {
	params := &LKADOTParams{SecLevel: secLevel, NumClients: numClients,
		VecLen: vecLen, BoundX: boundX, BoundY: boundY}
	return &LKADOT{Params: params}
}

// NewLKADOTFromParams takes configuration parameters of an existing
// LKADOT scheme instance, and reconstructs the scheme with the same
// configuration parameters. It returns a new LKADOT instance.
func NewLKADOTFromParams(params *LKADOTParams) *LKADOT {
	return &LKADOT{
		Params: params,
	}
}

// GenerateKeys generates a pair of master secret key and public key
// for the scheme. It returns an error in case keys could not be
// generated.
func (f LKADOT) GenerateKeys() (*LKADOTSecKey, *bn256.GT, error) {
	//sample keys for PRF (AES)
	prfKey := make([][]byte, f.Params.NumClients)

	for i := 0; i < f.Params.NumClients; i++ {
		prfKey[i] = make([]byte, 32)
		rand.Read(prfKey[i])
	}

	sampler := sample.NewUniformRange(big.NewInt(1), bn256.Order)
	mu, err := sampler.Sample()

	if err != nil {
		return nil, nil, err
	}
	gTMu := new(bn256.GT).ScalarBaseMult(mu)

	B := make([]data.Matrix, f.Params.NumClients)
	BStar := make([]data.Matrix, f.Params.NumClients)
	for i := 0; i < f.Params.NumClients; i++ {
		B[i], BStar[i], err = randomOB3(2*f.Params.VecLen+2*f.Params.SecLevel+2, mu)
		if err != nil {
			return nil, nil, err
		}
	}

	BHat := make([]data.Matrix, f.Params.NumClients)
	BStarHat := make([]data.Matrix, f.Params.NumClients)
	for i := 0; i < f.Params.NumClients; i++ {
		BHat[i] = make(data.Matrix, f.Params.VecLen+f.Params.SecLevel+2)
		BStarHat[i] = make(data.Matrix, f.Params.VecLen+f.Params.SecLevel+1)
		for j := 0; j < f.Params.VecLen+f.Params.SecLevel+1; j++ {
			if j < f.Params.VecLen {
				BHat[i][j] = B[i][j]
				BStarHat[i][j] = BStar[i][j]
			} else if j == f.Params.VecLen {
				BHat[i][j] = B[i][j+f.Params.VecLen]
				BStarHat[i][j] = BStar[i][j+f.Params.VecLen]
			} else if j < f.Params.VecLen+f.Params.SecLevel {
				BHat[i][j] = B[i][j-1+f.Params.VecLen+f.Params.SecLevel]
				BStarHat[i][j] = BStar[i][j+f.Params.VecLen]
			} else {
				BHat[i][j] = B[i][j-1+f.Params.VecLen+f.Params.SecLevel]
			}
		}
		// set tag slot at last value
		BHat[i][len(BHat[i])-1] = B[i][2*f.Params.VecLen+2*f.Params.SecLevel+1]
		BStarHat[i][len(BStarHat[i])-1] = BStar[i][2*f.Params.VecLen+2*f.Params.SecLevel+1]
	}
	return &LKADOTSecKey{BHat: BHat, BStarHat: BStarHat, PRFkey: prfKey}, gTMu, nil
}

// randomOB2 is a helping function that samples a random l x l matrix B
// and calculates BStar = mu * (B^-1)^T
func randomOB3(l int, mu *big.Int) (data.Matrix, data.Matrix, error) {
	sampler := sample.NewUniform(bn256.Order)
	B, err := data.NewRandomMatrix(l, l, sampler)
	if err != nil {
		return nil, nil, err
	}

	BStar, _, err := B.InverseModGauss(bn256.Order)
	if err != nil {
		return nil, nil, err
	}
	BStar = BStar.Transpose()
	BStar = BStar.MulScalar(mu)
	BStar = BStar.Mod(bn256.Order)

	return B, BStar, nil
}

// DeriveKey takes a matrix y whose rows are input vector y_1,...,y_m and
// master secret key, and returns the functional encryption key. That is
// a key that for encrypted x_1,...,x_m allows to calculate the sum of
// inner products <x_1,y_1> + ... + <x_m, y_m>. In case the key could not
// be derived, it returns an error.
func (f LKADOT) DeriveKey(y data.Matrix, secKey *LKADOTSecKey, c *big.Int, label []byte) (data.MatrixG2, error) {

	//generate tags
	tag := make([]*big.Int, f.Params.NumClients)
	var err error
	for i := 0; i < f.Params.NumClients; i++ {
		tag[i], err = generateTag(secKey.PRFkey[i], label)

		if err != nil {
			return nil, err
		}
	}

	// sample rho
	sampler := sample.NewUniform(bn256.Order)
	rho, err := data.NewRandomVector(f.Params.NumClients, sampler)
	if err != nil {
		return nil, err
	}

	// compute sum
	sumTags, err := rho.Dot(tag)
	if err != nil {
		return nil, err
	}

	gamma, err := data.NewRandomMatrix(f.Params.SecLevel, f.Params.NumClients, sampler)
	if err != nil {
		return nil, err
	}

	ones := data.NewConstantVector(f.Params.NumClients-1, big.NewInt(1))
	//random vector generatet from gamma
	r := data.NewVector(gamma[0][0:(f.Params.NumClients - 1)])
	//dotproduct of r and ones
	sum, err := r.Dot(ones)
	if err != nil {
		return nil, err
	}

	//compute modulus of sum, negate it add constant value to it
	sum.Neg(sum).Mod(sum, bn256.Order)
	sum.Add(sum, c)
	sum.Sub(sum, sumTags)
	sum.Mod(sum, bn256.Order)
	gamma[0][f.Params.NumClients-1] = sum

	zeros := data.NewConstantVector(2*f.Params.VecLen+2*f.Params.SecLevel+2, big.NewInt(0))
	keyMat := make(data.Matrix, f.Params.NumClients)
	var s *big.Int
	for i := 0; i < f.Params.NumClients; i++ {
		keyMat[i] = zeros.Copy()
		for j := 0; j < f.Params.VecLen+f.Params.SecLevel; j++ {
			if j < f.Params.VecLen {
				s = y[i][j]
			} else {
				s = gamma[j-f.Params.VecLen][i]
			}

			keyMat[i] = keyMat[i].Add(secKey.BStarHat[i][j].MulScalar(s))
			keyMat[i] = keyMat[i].Mod(bn256.Order)
		}
		//add rho to last slot
		keyMat[i] = keyMat[i].Add(secKey.BStarHat[i][len(secKey.BStarHat[i])-1].MulScalar(rho[i]))
		keyMat[i] = keyMat[i].Mod(bn256.Order)
	}

	return keyMat.MulG2(), nil
}

// Generate Tag from PRF based on AES
func generateTag(key []byte, label []byte) (*big.Int, error) {

	var err error
	//generate and initialize PRF
	c, err := aes.NewCipher(key)
	if err != nil {
		fmt.Println("error creating aes block cipher", err)
		return nil, err
	}
	stream := cipher.NewOFB(c, label)

	ks := make([]byte, 16)
	stream.XORKeyStream(ks, ks)
	tag := new(big.Int).Add(big.NewInt(0), new(big.Int).SetBytes(ks))

	return tag, nil

}

// Encrypt encrypts input vector x with the provided part of the master secret key, associated with a  label.
// It returns a ciphertext vector. If encryption failed, error is returned.
func (f LKADOT) Encrypt(x data.Vector, partSecKey data.Matrix, prfKey []byte, label []byte) (data.VectorG1, error) {

	tag, err := generateTag(prfKey, label)
	if err != nil {
		return nil, err
	}

	sampler := sample.NewUniform(bn256.Order)
	phi, err := data.NewRandomVector(f.Params.SecLevel, sampler)
	if err != nil {
		return nil, err
	}

	keyVec := data.NewConstantVector(2*f.Params.VecLen+2*f.Params.SecLevel+2, big.NewInt(0))
	var s *big.Int
	for j := 0; j < f.Params.VecLen+f.Params.SecLevel+1; j++ {
		if j < f.Params.VecLen {
			s = x[j]
		} else if j == f.Params.VecLen {
			s = big.NewInt(1)
		} else {
			s = phi[j-f.Params.VecLen-1]
		}

		keyVec = keyVec.Add(partSecKey[j].MulScalar(s))
		keyVec = keyVec.Mod(bn256.Order)
	}
	//add tag to last slot
	keyVec = keyVec.Add(partSecKey[len(partSecKey)-1].MulScalar(tag))
	keyVec = keyVec.Mod(bn256.Order)

	return keyVec.MulG1(), nil
}

// Decrypt accepts the ciphertext as a matrix whose rows are encryptions of vectors
// x_1,...,x_m and a functional encryption key corresponding to vectors y_1,...,y_m.
// It returns the sum of inner products <x_1,y_1> + ... + <x_m, y_m>. If decryption
// failed, an error is returned.
func (f *LKADOT) Decrypt(cipher data.MatrixG1, key data.MatrixG2, pubKey *bn256.GT) (*big.Int, error) {
	sum := f.DecryptWOSearch(cipher, key, pubKey)

	boundXY := new(big.Int).Mul(f.Params.BoundX, f.Params.BoundY)
	bound := new(big.Int).Mul(big.NewInt(int64(f.Params.NumClients*f.Params.VecLen)), boundXY)

	dec, err := NewCalc().InBN256().WithNeg().WithBound(bound).BabyStepGiantStep(sum, pubKey)

	return dec, err
	//return nil, nil
}

// Performs the decryption without the final search step
func (f *LKADOT) DecryptWOSearch(cipher data.MatrixG1, key data.MatrixG2, pubKey *bn256.GT) *bn256.GT {
	sum := new(bn256.GT).ScalarBaseMult(big.NewInt(0))
	for i := 0; i < f.Params.NumClients; i++ {
		for j := 0; j < 2*f.Params.VecLen+2*f.Params.SecLevel+2; j++ {
			paired := bn256.Pair(cipher[i][j], key[i][j])
			sum.Add(paired, sum)
		}
	}

	return sum
}
