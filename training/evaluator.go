package main

import (
	"fmt"
	"math"
	"math/big"
	"sync"
	"time"

	"github.com/JasZal/gofe/quadratic/noisy"
	"github.com/fentec-project/bn256"
)

type Evaluator struct {
	attr      int
	numClient int
	scaling   int
	pubKey    *bn256.GT
	fe        *noisy.SMNH
	cts       []*noisy.SMNHCT
	epsilon   float64
	delta     float64
	a         *Authority
}

func NewEvaluator(attr, numC, scaling int, cts []*noisy.SMNHCT, a *Authority, eps, d float64) *Evaluator {
	e := &Evaluator{
		attr:      attr,
		numClient: numC,
		pubKey:    a.getPP(),
		fe:        a.fe,
		cts:       cts,
		a:         a,
		epsilon:   eps,
		delta:     d,
		scaling:   scaling,
	}

	return e
}

func (e Evaluator) training(iterations int, batchsize int, alpha float64, logReg bool, boundR float64, nrWorkers int) ([]float64, time.Duration, error) {
	start := time.Now()

	theta := make([]float64, e.attr+1)
	// for i := 0; i < len(theta); i++ {
	// 	theta[i] = big.NewInt(0)
	// }

	len_rec := (e.numClient - 1) / (e.attr + 1)

	//todo anpassen?

	boundRes := big.NewInt(int64(math.Pow(float64(e.scaling), 3) * boundR))
	eT := 0.0
	//for i := 0; i < iterations; i++ {
	fmt.Println("***************************")
	for i := 0; i < iterations; i++ {

		eps := e.epsilon*math.Pow(float64(i+1)/float64(iterations), 1.5) - e.epsilon*math.Pow(float64(i)/float64(iterations), 1.5)
		eT += eps
		del := e.delta / float64(iterations)
		start2 := time.Now()

		bstart := (i * batchsize) % (len_rec)
		bend := (((i + 1) * batchsize) - 1) % (len_rec)

		t := time.Now()
		dk := e.a.generateDK(theta, bstart, bend, e.attr, eps, del, alpha, logReg, nrWorkers)
		fmt.Println("time generating DK: ", time.Since(t))

		var wg sync.WaitGroup

		chIn := make(chan int)

		// if i == 2 {
		// 	nrWorkers = 20
		// }
		// if i == 5 {
		// 	nrWorkers = 10
		// }
		// if i == 10 {
		// 	nrWorkers = 5
		// }
		// fmt.Printf("Iteration %v, worker %v\n", i, nrWorkers)

		for j := 0; j < nrWorkers; j++ {
			wg.Add(1)
			go e.evaluate(dk, theta, boundRes, chIn, &wg)
		}

		for j := 0; j < e.attr+1; j++ {
			chIn <- j
		}

		close(chIn)
		wg.Wait()

		timeI := time.Since(start2)
		fmt.Printf("theta^%v: %v time: %v\n", i, theta, timeI)

		UNUSED(t)
	}

	timeGD := time.Since(start)
	fmt.Printf("time whole GD: %v\n", timeGD)
	fmt.Println("***************************")
	return theta, timeGD, nil

}

func (e *Evaluator) evaluate(dk []*noisy.SMNHDK, theta []float64, b *big.Int, chIn chan int, wg *sync.WaitGroup) {

	defer wg.Done()
	for i := range chIn {

		f := noisy.NewSMNHFromParams(e.fe.Params)

		res, err := f.Decrypt(e.cts, dk[i], b, e.pubKey)

		if err != nil {
			fmt.Println("error at i :", i)
			//log.Fatal(err)
			return
		}

		theta[i] = float64(res.Int64()) / math.Pow(float64(e.a.scaling), 3)
		//fmt.Printf("theta[%v]: %v\n", i, theta[i])
	}
}
