package main

import (
	"QUILT/schemes"
	"fmt"
	"log"
	"math"
	"math/big"
	"runtime"
	"sync"
	"time"

	"github.com/JasZal/gofe/data"
	"github.com/google/differential-privacy/go/noise"
)

type Authority struct {
	secLevel int
	m        int
	n        int
	boundT   *big.Int
	pubKey   *schemes.OTNMCFEPP
	encKeys  []schemes.OTNMCFEEncKey
	msk      *schemes.OTNMCFESecKey
	fe       *schemes.OTNMCFE
	epsilon  float64
	delta    float64
	scaling  int64
}

func NewAuthority(secL int, m int, n int, bT *big.Int, e, d float64, scal int64) (*Authority, time.Duration) {
	a := &Authority{
		secLevel: secL,
		m:        m,
		n:        n,
		boundT:   bT,
		epsilon:  e,
		delta:    d,
		scaling:  scal,
	}

	var err error
	start := time.Now()
	a.fe = schemes.NewOTNMCFE(a.secLevel, a.n, a.m, nil, nil, nil, bT)
	a.msk, a.encKeys, a.pubKey, err = a.fe.GenerateKeys()
	timeSetup := time.Since(start)
	if err != nil {
		fmt.Println(err)
	}

	return a, timeSetup
}

// computes the InfSensitivity of a log Reg
func computeInfSen(theta []float64, alpha, scaling, numClients float64) float64 {
	sum := 0.0
	for i := 0; i < len(theta); i++ {
		sum += math.Abs(theta[i])
	}
	return (alpha / numClients) * (1 + 0.25*sum)

}

// computes the zeroSensitivity of a log Reg
func compute0Sen(theta []float64) int64 {
	return int64(len(theta))
}

// samples Noise according to requested function
func computeNoise(theta []float64, alpha, scaling, eps, del, numClients float64) []float64 {
	n := make([]float64, len(theta))

	infSen := computeInfSen(theta, alpha, scaling, numClients)
	zeroSen := compute0Sen(theta)

	//noise via gauss

	gauss := noise.Gaussian()
	for j := 0; j < len(n); j++ {
		n[j] = gauss.AddNoiseFloat64(0.0, zeroSen, infSen, eps, del)

	}

	//debug(fmt.Sprintf("n: %v\n", n))
	for j := 0; j < len(n); j++ {
		n[j] *= (math.Pow(scaling, 3))

		if n[j] >= 0 {
			n[j] = math.Ceil(n[j])
		} else {
			n[j] = math.Floor(n[j])
		}
	}

	return n
}

func (a *Authority) generateFunctionKey(yQuad [][]data.Matrix, yLin data.Matrix, yCon *big.Int, noise *big.Int, label []byte) (*schemes.OTNMCFEDecKey, error) {

	// derive a functional key for vector y
	key, err := a.fe.DeriveKey(yQuad, yLin, yCon, noise, label, a.msk)
	if err != nil {
		fmt.Println("Error during key derivation:", err)
	}

	return key, nil
}

func (a Authority) generateDK(theta []float64, attr int, numRec, eps, del, alpha float64, label []byte) ([]*schemes.OTNMCFEDecKey, [][][]data.Matrix) {
	// generate inner product vectors and put them in a matrix

	cols := attr + 1
	chunkCount := int(math.Ceil(float64(cols) / float64(a.m)))
	dk := make([]*schemes.OTNMCFEDecKey, cols)

	//check if key is permitted

	//check privacy budget
	nu := computeNoise(theta, alpha, float64(a.scaling), eps, del, float64(numRec))

	var quad [][][]data.Matrix
	var err error
	for j := 0; j < attr; j++ {
		yQuad := make([][]data.Matrix, a.n)
		yLin := data.NewConstantMatrix(a.n, a.m, big.NewInt(0))

		//const. value: c[0000] = theta[j]
		yCon := big.NewInt(int64(math.Round(theta[j] * math.Pow(float64(a.scaling), 3))))

		//initialize yQuad
		for i := 0; i < a.n; i++ {
			yQuad[i] = make([]data.Matrix, a.m)
			for k := 0; k < a.m; k++ {
				yQuad[i][k] = data.NewConstantMatrix(a.n, a.m, big.NewInt(0))
			}
		}

		wg := sync.WaitGroup{}

		chIn := make(chan int)

		for l := 0; l < runtime.NumCPU(); l++ {
			wg.Add(1)
			go func(chIn chan int) {
				defer wg.Done()
				for i := range chIn {

					// linear term  -alpha * (2 + theta[attr]) / (4*numRec) * x_i[j]
					yH := (-1 * alpha * (2 + theta[attr])) / (4.0 * numRec)
					yLin[i*chunkCount+j/a.m][j%a.m] = big.NewInt(int64(math.Round(yH * float64(a.scaling) * float64(a.scaling))))

					//set quadratic terms
					//alpha/numRec*x_i[attr]x_i[j]
					yH = alpha / numRec
					yQuad[i*chunkCount+attr/a.m][attr%a.m][i*chunkCount+j/a.m][j%a.m] = big.NewInt(int64(math.Round(yH * float64(a.scaling))))
					//c[i*cols+j+1][0][(i+1)*cols][0] = big.NewInt(int64(math.Round(yH * float64(a.scaling))))

					for k := 0; k < attr; k++ {
						//-alpha*Thetak/4numRec * x_i[k]x_i[j]
						yH = (-1 * theta[k] * alpha) / (4 * numRec)
						//if k <= j {
						yQuad[i*chunkCount+k/a.m][k%a.m][i*chunkCount+j/a.m][j%a.m] = big.NewInt(int64(math.Round(yH * float64(a.scaling))))
						//c[i*cols+k+1][0][i*cols+j+1][0] = big.NewInt(int64(math.Round(yH * float64(a.scaling))))
						//} else {
						//	yQuad[i*chunkCount+j/a.m][j%a.m][i*chunkCount+k/a.m][k%a.m] = big.NewInt(int64(math.Round(yH * float64(a.scaling))))
						//c[i*cols+j+1][0][i*cols+k+1][0] = big.NewInt(int64(math.Round(yH * float64(a.scaling))))

						//}
					}
				}
			}(chIn)

		}
		for l := 0; l < int(numRec); l++ {
			chIn <- l
		}

		close(chIn)
		wg.Wait()

		quad = append(quad, yQuad)
		//start3 := time.Now()
		dk[j], err = a.generateFunctionKey(yQuad, yLin, yCon, big.NewInt(int64(nu[j])), label)
		//debug(fmt.Sprintf("time to generate dk[%v] : %v \n", j, time.Since(start3)))
		if err != nil {
			log.Fatal("Error during Function Key Derivation:", err)
		}
	}

	//bias term
	yQuad := make([][]data.Matrix, a.n)
	yLin := data.NewConstantMatrix(a.n, a.m, big.NewInt(0))
	//theta[attr]-alpha/numRec - alpha*theta[attr]/3*numRec
	yH := theta[attr] - alpha/2 - (alpha*theta[attr])/(4)
	yCon := big.NewInt(int64(math.Round(yH * float64(a.scaling) * float64(a.scaling) * float64(a.scaling))))
	//initialize yQuad
	for i := 0; i < a.n; i++ {
		yQuad[i] = make([]data.Matrix, a.m)
		for k := 0; k < a.m; k++ {
			yQuad[i][k] = data.NewConstantMatrix(a.n, a.m, big.NewInt(0))
		}
	}

	wg := sync.WaitGroup{}

	chIn := make(chan int)

	for i := 0; i < runtime.NumCPU(); i++ {
		wg.Add(1)
		go func(chIn chan int) {
			defer wg.Done()
			for i := range chIn {
				// linear term  alpha *  / (numRec) * x_i[attr]
				yH := alpha / numRec
				yLin[i*chunkCount+attr/a.m][attr%a.m] = big.NewInt(int64(math.Round(yH * float64(a.scaling) * float64(a.scaling))))

				for k := 0; k < attr; k++ {
					//-alpha*Thetak/4numRec * x_i[k]x_i[j]
					yH = (-1 * theta[k] * alpha) / (4 * numRec)
					yLin[i*chunkCount+k/a.m][k%a.m] = big.NewInt(int64(math.Round(yH * float64(a.scaling) * float64(a.scaling))))
				}
			}
		}(chIn)
	}
	for i := 0; i < int(numRec); i++ {
		chIn <- i
	}

	close(chIn)
	wg.Wait()

	quad = append(quad, yQuad)
	dk[attr], err = a.generateFunctionKey(yQuad, yLin, yCon, big.NewInt(int64(nu[attr])), label)
	if err != nil {
		log.Fatal("Error during Function Key Derivation:", err)
	}

	return dk, quad

}

func (a Authority) getEncryptionKey(pos int) schemes.OTNMCFEEncKey {
	return a.encKeys[pos]
}

func (a Authority) getParams() *schemes.OTNMCFEParams {
	return a.fe.Params
}

func (a Authority) getPP() *schemes.OTNMCFEPP {
	return a.pubKey
}
