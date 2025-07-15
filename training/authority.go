package main

import (
	"fmt"
	"log"
	"math"
	"math/big"
	"sync"
	"time"

	"github.com/JasZal/gofe/data"
	"github.com/JasZal/gofe/quadratic/noisy"
	"github.com/fentec-project/bn256"
	"github.com/google/differential-privacy/go/noise"
)

type Authority struct {
	secLevel  int
	vecLen    int
	numClient int
	boundX    *big.Int
	boundY    *big.Int
	boundN    *big.Int
	pubKey    *bn256.GT
	encKeys   []*noisy.SMNHEncKey
	msk       *noisy.SMNHSecKey
	fe        *noisy.SMNH
	epsilon   float64
	delta     float64
	scaling   int64
	ymax      float64
}

func NewAuthority(secL int, vecL int, numC int, bX, bY, bN *big.Int, e, d float64, scal int64, y float64) (*Authority, time.Duration) {
	a := &Authority{
		secLevel:  secL,
		vecLen:    vecL,
		numClient: numC,
		boundX:    bX,
		boundY:    bY,
		boundN:    bN,
		epsilon:   e,
		delta:     d,
		scaling:   scal,
		ymax:      y,
	}

	var err error
	start := time.Now()
	a.fe = noisy.NewSMNH(a.secLevel, a.numClient, a.vecLen, a.boundX, a.boundY, a.boundN)
	timeSetup := time.Since(start)
	a.msk, a.encKeys, a.pubKey, err = a.fe.GenerateKeys()
	if err != nil {
		fmt.Println(err)
	}

	return a, timeSetup
}

func computeInfSen(theta []float64, alpha, b, scaling, maxy float64, logReg bool) float64 {
	sum := 0.0
	for i := 0; i < len(theta); i++ {
		sum += math.Abs(theta[i])
	}
	if logReg {
		//logistic Regression
		return alpha / b * (1 + 0.25*sum)
	} else {
		//Linear Regression

		return (alpha*maxy)/b + (alpha/b)*sum

	}

}

func compute0Sen(theta []float64) int64 {
	return int64(len(theta))
}

func computeNoise(theta []float64, alpha, b, scaling, maxy, eps, del float64, logReg bool) []float64 {
	n := make([]float64, len(theta))

	infSen := computeInfSen(theta, alpha, b, scaling, maxy, logReg)
	zeroSen := compute0Sen(theta)

	//noise via gauss

	gauss := noise.Gaussian()
	for j := 0; j < len(n); j++ {
		n[j] = gauss.AddNoiseFloat64(0.0, zeroSen, infSen, eps, del)

	}

	//fmt.Printf("n: %v\n", n)

	for j := 0; j < len(n); j++ {
		n[j] *= (math.Pow(scaling, 3))

		if n[j] >= 0 {
			n[j] = math.Ceil(n[j])
		} else {
			n[j] = math.Floor(n[j])
		}
	}

	//n = make([]float64, len(theta))

	return n
}

func (a *Authority) generateFunctionKey(y [][]data.Matrix, noise *big.Int) (*noisy.SMNHDK, error) {

	// derive a functional key for vector y
	key, err := a.fe.DeriveKey(y, noise, a.msk)
	if err != nil {
		fmt.Println("Error during key derivation:", err)
	}

	return key, nil
}

func (a Authority) generateDK(theta []float64, bstart, bend, attr int, eps, del, alpha float64, logReg bool, nrWorkers int) []*noisy.SMNHDK {
	// generate inner product vectors and put them in a matrix

	cols := attr + 1
	b := float64(bend - bstart + 1)
	dk := make([]*noisy.SMNHDK, cols)

	//check if key is permitted

	//check privacy budget

	nu := computeNoise(theta, alpha, b, float64(a.scaling), a.ymax, eps, del, logReg)

	var err error
	for j := 0; j < attr; j++ {

		c := make([][]data.Matrix, a.numClient)
		for i := 0; i < a.numClient; i++ {
			c[i] = make([]data.Matrix, a.vecLen)
			for k := 0; k < a.vecLen; k++ {
				c[i][k] = data.NewConstantMatrix(a.numClient, a.vecLen, big.NewInt(0))
			}
		}

		wg := sync.WaitGroup{}

		chIn := make(chan int)

		for i := 0; i < nrWorkers; i++ {
			wg.Add(1)
			go func(chIn chan int) {
				defer wg.Done()
				for i := range chIn {

					yH := alpha / b
					c[i*cols+j+1][0][(i+1)*cols][0] = big.NewInt(int64(math.Round(yH * float64(a.scaling))))

					for k := 0; k < attr; k++ {
						yH = -1 * theta[k] * alpha / b
						if logReg {
							//logistic Regression
							yH *= 1.0 / 4.0
						}
						if k <= j {
							c[i*cols+k+1][0][i*cols+j+1][0] = big.NewInt(int64(math.Round(yH * float64(a.scaling))))
						} else {
							c[i*cols+j+1][0][i*cols+k+1][0] = big.NewInt(int64(math.Round(yH * float64(a.scaling))))

						}
					}
					yH = -1 * theta[attr] * alpha / b
					if logReg {
						//logistic Regression
						yH = -1 * alpha * (2 + theta[attr]) / (4.0 * b)
					}
					c[0][0][i*cols+j+1][0] = big.NewInt(int64(math.Round(yH * float64(a.scaling))))
				}
			}(chIn)

		}

		for i := bstart; i < bend; i++ {
			chIn <- i
		}

		close(chIn)
		wg.Wait()

		c[0][0][0][0] = big.NewInt(int64(math.Round(theta[j] * float64(a.scaling))))

		dk[j], err = a.generateFunctionKey(c, big.NewInt(int64(nu[j])))
		if err != nil {
			log.Fatal("Error during Function Key Derivation:", err)
		}
	}

	c := make([][]data.Matrix, a.numClient)
	for i := 0; i < a.numClient; i++ {
		c[i] = make([]data.Matrix, a.vecLen)
		for j := 0; j < a.vecLen; j++ {
			c[i][j] = data.NewConstantMatrix(a.numClient, a.vecLen, big.NewInt(0))
		}
	}

	//bias term
	wg := sync.WaitGroup{}

	chIn := make(chan int)

	for i := 0; i < nrWorkers; i++ {
		wg.Add(1)
		go func(chIn chan int) {
			defer wg.Done()
			for i := range chIn {

				yH := alpha / b

				c[0][0][(i+1)*(cols)][0] = big.NewInt(int64(math.Round(yH * float64(a.scaling))))
				for k := 0; k < attr; k++ {

					yH = -1 * theta[k] * alpha / b

					if logReg {
						//logistic Regression
						yH *= 1.0 / 4.0
					}

					c[0][0][i*cols+k+1][0] = big.NewInt(int64(math.Round(yH * float64(a.scaling))))

				}

			}
		}(chIn)

	}

	for i := bstart; i < bend; i++ {
		chIn <- i
	}

	close(chIn)
	wg.Wait()

	yH := theta[attr] - theta[attr]*alpha
	if logReg {
		//logistic Regression
		yH = theta[attr] - alpha/b - alpha*theta[attr]/(3*b)
	}
	z := big.NewInt(int64(math.Round(yH * float64(a.scaling))))
	c[0][0][0][0] = big.NewInt(int64(math.Round(yH * float64(a.scaling))))
	//fmt.Printf("c: %v\n", c)
	dk[attr], err = a.generateFunctionKey(c, big.NewInt(int64(nu[attr])))
	if err != nil {
		log.Fatal("Error during Function Key Derivation: ", err)
	}
	UNUSED(z)
	return dk
}

func (a Authority) getEncryptionKey(pos int) *noisy.SMNHEncKey {
	return a.encKeys[pos]
}

func (a Authority) getParams() *noisy.SMNHParams {
	return a.fe.Params
}

func (a Authority) getPP() *bn256.GT {
	return a.pubKey
}
