package main

import (
	"encoding/csv"
	"fmt"
	"log"
	"math"
	"math/big"
	"os"
	"strconv"
	"sync"
	"time"

	"github.com/JasZal/gofe/data"
	"github.com/JasZal/gofe/quadratic/noisy"
)

var deb bool = true

func write(filename string, message string) {

	var file *os.File
	var err error

	file, err = os.OpenFile(filename, os.O_RDWR|os.O_CREATE|os.O_APPEND, 0755)

	if err != nil {
		log.Fatal(err)
	}
	defer file.Close()

	_, err = file.WriteString(message)
	if err != nil {
		log.Fatal(err)
	}

}

func debug(s string) {
	if deb {
		fmt.Println(s)
	}
}

func UNUSED(x ...interface{}) {}
func main() {
	fileR := "results.txt"
	//attributes
	secLevel := 1
	s := []int{1000, 10000}
	iterations := 25

	ymax := 10.0
	for _, scaling := range s {

		write(fileR, fmt.Sprintf("%v :\n", scaling))

		boundX := big.NewInt(int64(ymax) * int64(scaling))
		boundY := big.NewInt(int64(10 * scaling))
		boundN := big.NewInt(int64(1 * math.Pow(float64(scaling), 5)))
		boundT := 10.0

		//read data
		prefix := "./datasets/training"
		files := []string{"LBW", "PCS", "UIS"}
		alphas := []float64{0.1, 0.3, 0.1}
		postfix := ".csv"

		//training := "./datasets/trainingCoronary10.csv"
		logReg := true

		for tI, training := range files {

			dataPlain, attr := loadData(prefix+training+postfix, scaling)
			n := len(dataPlain)
			batchsize := (n - 1) / (attr + 1)

			nrWorkers := 20

			alpha := alphas[tI]
			epsilon := 5.0
			delta := 1.0 / (float64(n-1) / float64((attr + 1)))

			if (n-1)/(attr+1)%batchsize != 0 {
				fmt.Println("batchsize not dividor of #recs")
			}

			fmt.Printf("attr: %v, label: 1, length: %v, batchsize: %v, scaling: %v\n", attr, (n-1)/(attr+1), batchsize, scaling)

			var err error

			//setup scheme/authority
			m := 1
			a, tSetupA := NewAuthority(secLevel, m, n, boundX, boundY, boundN, epsilon, delta, int64(scaling), ymax)
			if err != nil {
				log.Fatal("Error during Key Generation:  %v", err)
			}
			debug("time Setup: " + tSetupA.String())

			//setup clients/encrypt data
			ct := make([]*noisy.SMNHCT, n)
			wg := sync.WaitGroup{}

			chIn := make(chan int)
			startE := time.Now()

			for i := 0; i < nrWorkers; i++ {
				wg.Add(1)
				go func(chIn chan int) {
					defer wg.Done()
					for i := range chIn {
						client := noisy.NewSMNHFromParams(a.getParams())
						start := time.Now()
						ct[i], err = client.Encrypt(a.getEncryptionKey(i), data.NewConstantVector(1, dataPlain[i]))
						timeEnc := time.Since(start)
						if i == 0 {
							debug("time Enc one rec: " + timeEnc.String())
						}
					}
				}(chIn)

			}

			for i := 0; i < n; i++ {
				chIn <- i
			}

			close(chIn)
			wg.Wait()

			tE := time.Since(startE)
			fmt.Printf("time Encryption total: %v\n", tE)

			//setup evaluator
			e := NewEvaluator(attr, n, scaling, ct, a, epsilon, delta)
			//start training

			theta, tReg, err := e.training(iterations, batchsize, alpha, logReg, boundT, nrWorkers)
			if err != nil {
				fmt.Printf("Runtime: %v\n", tReg)
				log.Fatal("Error during Training:", err)
			}
			fmt.Printf("main: theta: %v\n", theta)
			fmt.Printf("main: time Reg: %v\n", tReg)
			write(fileR, fmt.Sprintf("%v : %v\n", training, tReg))
		}
	}
}

// data set vertical splitted, m = attr/2
func loadData(file string, scaling int) (data.Vector, int) {

	f, err := os.Open(file)
	if err != nil {
		log.Fatal(err)
	}
	defer f.Close()

	dataR := csv.NewReader(f)
	records, err := dataR.ReadAll()
	if err != nil {
		log.Fatal(err)
	}

	cols := len(records[0])

	dataX := data.NewConstantVector(1+len(records)*cols, big.NewInt(1*int64(scaling)))

	for i := 0; i < len(records); i++ {
		for j := 0; j < cols; j++ {
			interm, _ := strconv.ParseFloat(records[i][j], 64)
			interm = math.Round(interm * float64(scaling))
			dataX[i*(cols)+j+1] = big.NewInt(int64(interm))

		}
	}
	return dataX, cols - 1
}
