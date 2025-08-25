package main

import (
	"QUILT/schemes"
	"encoding/csv"
	"fmt"
	"log"
	"math"
	"math/big"
	"os"
	"runtime"
	"strconv"
	"sync"
	"time"

	"github.com/JasZal/gofe/data"
)

var deb bool = true

// writes given string to file
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

// prints given string if debugging is active
func debug(s string) {
	if deb {
		fmt.Print(s)
	}
}

// to handle unsued variables
func UNUSED(x ...interface{}) {}

// This method trains a logistic regrestion and stores results in the specified file
// Following variables can be set:
// s: 			range of scaling factor
// secLevel: 	secLevel of the underlying FE schemes
// iterations: 	number of training iterations
// epsilon: 		for DP (delta is set to 1/#rec) (does not have an affect on the runtime)
// splits:		determines how many vertical splits are assumed in the data, if split == 0 , this means a maximal splitted dataset
// rounds: how many rounds this should run
func main() {
	//file to save results
	fileR := "resultsAverage.txt"

	//attributes
	rounds := 1
	splits := 0 //describes how many vertical splits are assumed, 0 means maximal split

	s := []int{10000} //sets the scaling factor
	iterations := 25  //number of iterations for gradient descent algorithm
	epsilon := 5.0    //does not make a difference for runtime
	secLevel := 1     //describes the secLevel of the scheme

	if splits != 0 {
		write(fileR, fmt.Sprintf("%v splits \n", splits))
	} else {
		write(fileR, "maximal splitted\n")
	}

	debug(fmt.Sprintf("split: %v\n", splits))
	for _, scaling := range s {
		write(fileR, fmt.Sprintf("%v :\n", scaling))

		boundTheta := 10.0 //10.0
		boundT := big.NewInt(int64(boundTheta * math.Pow(float64(scaling), 3)))
		label := make([]byte, 16)

		//read data
		//label is always assumed to be in the last slot
		prefix := "./datasets/training"
		files := []string{"LBW", "PCS", "UIS"}
		alphas := []float64{0.1, 0.3, 0.1} //learning rate that is used for gradient descent per dataset

		postfix := ".csv"

		for tI, training := range files {

			dataPlain, m, attr := loadData(prefix+training+postfix, scaling, splits)
			n := len(dataPlain)
			var numRec int
			if splits != 0 {
				numRec = int(float64(n) / float64((splits + 1)))
			} else {
				numRec = int(float64(n) / float64((attr + 1)))
			}

			alpha := alphas[tI]
			delta := 1.0 / float64(numRec)

			debug(fmt.Sprintf("Dataset: %v, #rec: %v, #attr: %v, #label: 1, scaling: %v\n", training, numRec, attr, scaling))
			for r := 0; r < rounds; r++ {
				debug(fmt.Sprintf("round: %v \n", r))
				var err error

				//setup scheme/authority
				a, tSetupA := NewAuthority(secLevel, m, n, boundT, epsilon, delta, int64(scaling))
				if err != nil {
					log.Fatal("Error during Setup:  %v\n", err)
				}
				debug("time Setup: " + tSetupA.String())

				//setup clients/encrypt data
				ct := make([]*schemes.OTNMCFECT, n)
				wg := sync.WaitGroup{}

				chIn := make(chan int)
				startE := time.Now()

				for i := 0; i < runtime.NumCPU(); i++ {
					wg.Add(1)
					go func(chIn chan int) {
						defer wg.Done()
						for i := range chIn {
							client := schemes.NewOTNMCFEFromParams(a.getParams())
							start := time.Now()
							ct[i], err = client.Encrypt(a.getEncryptionKey(i), dataPlain[i], label)
							timeEnc := time.Since(start)
							if i == 0 {
								debug("time Enc one rec: " + timeEnc.String() + "\n")
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
				debug(fmt.Sprintf("time Encryption total: %v\n", tE))

				//setup evaluator
				e := NewEvaluator(attr, n, scaling, ct, a, epsilon, delta, label)
				//start training
				theta, timeLogReg, err := e.training(iterations, numRec, alpha, boundT)
				if err != nil {
					debug(fmt.Sprintf("Runtime: %v\n", timeLogReg))
					log.Fatal("Error during Training:", err)
				}

				debug(fmt.Sprintf("***************************"))
				debug(fmt.Sprintf("main: final theta: %v\n", theta))
				debug(fmt.Sprintf("main: final time LogReg: %v\n", timeLogReg))
				write(fileR, fmt.Sprintf("%v : %v\n", training, timeLogReg))

			}
		}

	}

}

//	takes the file and the scaling factor and returns a dataVector with all data points and the number of attributes (col-1)
//
// data is splitted vertically in splits vectors. if split = 0 this means a maximal split dataset is assumed
func loadData(file string, scaling, splits int) (data.Matrix, int, int) {

	//load data
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

	var m int
	if splits != 0 {
		m = int(math.Ceil(float64(len(records[0])) / float64(splits+1)))
	} else {
		m = 1
	}

	//determine how many chunks per row are neccesary
	chunkCount := int(math.Ceil(float64(len(records[0])) / float64(m)))

	//save the chunks as data.Vectors in a slice
	var dataX data.Matrix
	for i := 0; i < len(records); i++ {
		dataRow := data.NewConstantMatrix(chunkCount, m, big.NewInt(0))
		for j := 0; j < len(records[i]); j++ {
			interm, err := strconv.ParseFloat(records[i][j], 64)
			if err != nil {
				log.Fatal(err)
			}
			interm = math.Round(interm * float64(scaling))
			dataRow[int(math.Floor(float64(j)/float64(m)))][j%m] = big.NewInt(int64(interm))
		}
		for j := 0; j < chunkCount; j++ {
			dataX = append(dataX, dataRow[j])
		}

	}

	return dataX, m, len(records[0]) - 1

}
