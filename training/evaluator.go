package main

import (
	"QUILT/schemes"
	"fmt"
	"math"
	"math/big"
	"runtime"
	"sync"
	"time"

	"github.com/JasZal/gofe/data"
)

type Evaluator struct {
	attr    int
	n       int
	scaling int
	pubKey  *schemes.OTNMCFEPP
	fe      *schemes.OTNMCFE
	cts     []*schemes.OTNMCFECT
	epsilon float64
	delta   float64
	a       *Authority
	label   []byte
}

func NewEvaluator(attr, n, scaling int, cts []*schemes.OTNMCFECT, a *Authority, eps, d float64, label []byte) *Evaluator {
	e := &Evaluator{
		attr:    attr,
		n:       n,
		pubKey:  a.getPP(),
		fe:      a.fe,
		cts:     cts,
		a:       a,
		epsilon: eps,
		delta:   d,
		scaling: scaling,
		label:   label,
	}

	return e
}

func (e Evaluator) training(iterations, numRec int, alpha float64, boundR *big.Int) ([]float64, time.Duration, error) {

	start := time.Now()

	theta := make([]float64, e.attr+1)

	boundRes := new(big.Int).Mul(boundR, big.NewInt(int64(math.Pow(float64(e.scaling), 3))))
	epsTotal := 0.0
	debug(fmt.Sprintln("************Training***************"))

	for i := 0; i < iterations; i++ {

		eps := e.epsilon*math.Pow(float64(i+1)/float64(iterations), 1.5) - e.epsilon*math.Pow(float64(i)/float64(iterations), 1.5)
		epsTotal += eps
		del := e.delta / float64(iterations)

		start2 := time.Now()

		t := time.Now()
		dk, yQuad := e.a.generateDK(theta, e.attr, float64(numRec), eps, del, alpha, e.label)

		fmt.Println("time generating DK: ", time.Since(t))

		var wg sync.WaitGroup

		chIn := make(chan int)

		for j := 0; j < runtime.NumCPU(); j++ {
			wg.Add(1)
			go e.evaluate(dk, yQuad, theta, boundRes, chIn, &wg)
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

func (e *Evaluator) evaluate(dk []*schemes.OTNMCFEDecKey, yQuad [][][]data.Matrix, theta []float64, b *big.Int, chIn chan int, wg *sync.WaitGroup) {

	defer wg.Done()
	for i := range chIn {

		res, err := e.fe.Decrypt(dk[i], yQuad[i], e.cts, e.pubKey)

		if err != nil {
			fmt.Println("error at i :", i)
			//log.Fatal(err)
			return
		}

		theta[i] = float64(res.Int64()) / math.Pow(float64(e.a.scaling), 3)
		fmt.Printf("theta[%v]: %v\n", i, theta[i])
	}
}
