package main

import (
	"fmt"
	"log"
	"math"
	"math/big"
	"os"
	"runtime"
	"sync"
	"time"

	"github.com/JasZal/gofe/data"
	"github.com/JasZal/gofe/quadratic/noisy"
	"github.com/JasZal/gofe/sample"
	"github.com/fentec-project/bn256"
)

var coeffManipulation bool
var logSearch bool

// Performs benchmarking for ZSHA Scheme (ZSHA) and new construction (QUILT)
// The schemes are evaluated for all algorithms: setup, encryption, keygen and decryption.
// You can choose the parameters of m and the x-axis, as well if decryption should be performed on sparse functions or arbitrary functions and with or without final search step
// Plaintexts and FunctionCoefficients are sampled randomly from the given range.
func main() {

	t := time.Now()
	rounds := 1.0

	fileZSHA := "results/ZSHA" + t.Format("02_01_15:04") + ".txt"
	fileQUILT := "results/QUILT_" + t.Format("02_01_15:04") + ".txt"

	files := []string{fileZSHA, fileQUILT}

	debug := true
	// choose the parameters for the scheme
	secLevel := 1
	boundX := big.NewInt(128)
	boundY := big.NewInt(128)
	boundN := big.NewInt(10)
	sizes := [][]int{}

	//choose if decryption is performed with or without the final search step
	logSearch = false

	//describes how many coefficients are != 0 in the polynomial
	coeffManipulation = false //describes if random functions with arbitrary coefficients are chosen or if the number is manipulated
	minCoeffN0 := 0           //min #coeff != 0
	maxCoeffN0 := 200         //max #coeff != 0, only necessary if coeffMan. = true

	//incremetion step counter:
	cCoeff := 50 //incremention step for #coeff != 0
	cX := 5.0    //50.0   //counter for x axis
	cj := 25     //counter for vectorlength

	//range of data set size
	minX := 0.0
	maxX := 1 //20000.0

	//set vectorlength range
	maxJ := 1 // 50
	for j := 0; j <= maxJ; j = j + cj {
		l := j
		if j == 0 {
			l = j + 1
		}
		if j >= 5 {
			cj = 20
		}
		if j >= 25 {
			cj = 25
		}

		//set k (num of clients) accordingly
		for i := int(math.Ceil(minX / float64(l))); i*l <= int(maxX)+int(math.Ceil(cX/float64(l))); i = i + int(math.Ceil(cX/float64(l))) {

			k := i
			if i == 0 {
				if l > int(cX) {
					continue
				}
				k = i + 1
			}

			if i*l >= 25 {
				cX = 10
			}
			if i*l >= 50 {
				cX = 100
			}

			if i*l >= 500 {
				cX = 500
			}
			// if i*l >= 1000 {
			// 	cX = 1000
			// }

			sizes = append(sizes, []int{k, l})
		}
	}

	if debug {
		fmt.Printf("sizes: %v\n", sizes)
	}

	//create table in paper
	//sizes = [][]int{{2, 1}, {2, 2}, {2, 3}, {10, 1}, {10, 2}, {10, 3}, {100, 1}, {100, 2}, {100, 1}}
	//logSearch = true

	count := 0

	for j := 0; j < len(files); j++ {

		if !logSearch {
			write(files[j], fmt.Sprintln("N, #numClient, #vecLen, #coeff!=0, Setup, Encryption, DeriveKey, DecryptionWOSearch Time in Nanoseconds"), false)
		} else {
			write(files[j], fmt.Sprintln("N, #numClient, #vecLen, #coeff!=0, Setup, Encryption, DeriveKey, Decryption Time in Nanoseconds"), false)
		}

	}

	for i := 0; i < len(sizes); i++ {

		if debug {
			fmt.Println("************************************* New Parameters ******************************")
		}

		// sample vectors that will be encrypted
		n := sizes[i][0]
		m := sizes[i][1]

		//cound maximal number of possible coefficients != 0
		numCoeff := 0
		for i := 0; i < n+1; i++ {
			for j := 0; j < m; j++ {
				for k := 0; k < n+1; k++ {
					for l := 0; l < m; l++ {
						if i < k || (i == k && j <= l) {
							numCoeff++
						}
					}
				}
			}
		}

		if !coeffManipulation {
			minCoeffN0 = numCoeff
			maxCoeffN0 = numCoeff
		} else {
			if debug {
				fmt.Printf("counter coeff != 0 maximal possible: %v\n", numCoeff)
				fmt.Printf("used  coeff != 0 range up to: %v\n", maxCoeffN0)

			}

		}

		for sparseLevel := minCoeffN0; sparseLevel <= int(math.Min(float64(maxCoeffN0), float64(numCoeff))); sparseLevel += cCoeff {
			if sparseLevel > 90 {
				cCoeff = 100
			}
			if sparseLevel > 1000 {
				cCoeff = 1000
			}

			var timeSetup, timeEnc, timeDK, timeDec time.Duration
			tS := make([][]float64, 4)
			for i := 0; i < len(tS); i++ {
				tS[i] = []float64{0.0, 0.0}
			}
			for j := 0; j < int(rounds); j++ {

				sampler := sample.NewUniformRange(new(big.Int).Add(new(big.Int).Neg(boundX), big.NewInt(1)), boundX)
				x := make(data.Matrix, n)

				for i := 0; i < n; i++ {
					x[i], _ = data.NewRandomVector(m, sampler)
				}

				if sizes[i][1] == 1 {
					timeSetup, timeEnc, timeDK, timeDec = runZSHA(secLevel, sizes[i][1], sizes[i][0]+1, sparseLevel, boundX, boundY, boundN, x, debug)

					count = 0
					tS[0][count] += float64(timeSetup.Nanoseconds())
					tS[1][count] += float64(timeEnc.Nanoseconds())
					tS[2][count] += float64(timeDK.Nanoseconds())
					tS[3][count] += float64(timeDec.Nanoseconds())

				}

				timeSetup, timeEnc, timeDK, timeDec = runQUILT(secLevel, sizes[i][1], sizes[i][0], sparseLevel, boundX, boundY, boundN, x, debug)

				count = 1
				tS[0][count] += float64(timeSetup.Nanoseconds())
				tS[1][count] += float64(timeEnc.Nanoseconds())
				tS[2][count] += float64(timeDK.Nanoseconds())
				tS[3][count] += float64(timeDec.Nanoseconds())

			}

			for j := 0; j < len(files); j++ {

				write(files[j], fmt.Sprintln(sizes[i][0]*sizes[i][1], ",", sizes[i][0], ",", sizes[i][1], ",", sparseLevel, ",", tS[0][j]/rounds, ",", float64(tS[1][j]/rounds), ",", tS[2][j]/rounds, ",", tS[3][j]/rounds, ";"), true)

			}

		}

	}

}

// runs the ZSHA scheme for given input parameters and outputs the needed time per algorithm
func runZSHA(secLevel, vecLen, numClient, sparseLevel int, boundX, boundY, boundN *big.Int, xH data.Matrix, debug bool) (time.Duration, time.Duration, time.Duration, time.Duration) {
	if debug {
		fmt.Println("******************ZSHA******************")
		fmt.Println("N:", numClient*vecLen, "numClients: ", numClient-1, " VecLen: ", vecLen, "sparseLevel:", sparseLevel)
		//fmt.Println("achieved true quadratic functionality by setting m=m+1")
	}

	x := data.NewConstantMatrix(numClient, vecLen, big.NewInt(1))

	for i := 0; i < len(xH); i++ {
		x[i+1] = xH[i]

	}

	// build the scheme
	fe := noisy.NewSMNH(secLevel, numClient, vecLen, boundX, boundY, boundN)

	// generate master secret key, encryption keys and public key
	start := time.Now()
	masterSecKey, enckeys, pubKey, _ := fe.GenerateKeys()
	timeSetup := time.Since(start)
	if debug {
		fmt.Println("time Setup: ", timeSetup)
	}

	// // Setup a channel for collecting results
	cipher := make([]*noisy.SMNHCT, numClient)
	var wg sync.WaitGroup
	var timeEnc time.Duration
	maxWorkers := runtime.NumCPU()
	sem := make(chan struct{}, maxWorkers)
	startT := time.Now()

	for i := 0; i < numClient; i++ {
		i := i
		wg.Add(1)

		go func() {
			defer wg.Done()
			sem <- struct{}{}        // acquire slot
			defer func() { <-sem }() // release slot

			if i == 0 {
				start = time.Now()
				cipher[i], _ = fe.Encrypt(enckeys[i], x[i])
				timeEnc = time.Since(start)

				if debug {
					fmt.Println("Time for one encryption:", timeEnc)
				}
			} else {
				// For other encryptions
				cipher[i], _ = fe.Encrypt(enckeys[i], x[i])

			}
		}()

	}
	wg.Wait()

	if debug {
		fmt.Println("Time encryption total:", time.Since(startT))
	}

	// sample random inner product vectors and put them in a matrix
	// sparselevel describes the number of coefficients != 0
	counter := 0
	sampler := sample.NewUniformRange(new(big.Int).Add(new(big.Int).Neg(boundY), big.NewInt(1)), boundY)
	c := make([][]data.Matrix, numClient)

	var counterMutex sync.Mutex

	for i := 0; i < numClient; i++ {
		i := i // capture loop variable
		wg.Add(1)
		go func() {
			defer wg.Done()
			sem <- struct{}{}        // acquire slot
			defer func() { <-sem }() // release slot

			c[i] = make([]data.Matrix, vecLen)
			for j := 0; j < vecLen; j++ {
				c[i][j] = data.NewConstantMatrix(numClient, vecLen, big.NewInt(0))
				for k := 0; k < numClient; k++ {
					for l := 0; l < vecLen; l++ {
						if i < k || (i == k && j <= l) {
							if !coeffManipulation {
								c[i][j][k][l], _ = sampler.Sample()
								c[i][j][k][l].Mod(c[i][j][k][l], bn256.Order)
								continue
							}
							counterMutex.Lock()
							if counter < sparseLevel {
								counter++
								counterMutex.Unlock()
								c[i][j][k][l], _ = sampler.Sample()
								c[i][j][k][l].Mod(c[i][j][k][l], bn256.Order)
								continue
							}
							counterMutex.Unlock()
						}
						c[i][j][k][l] = big.NewInt(0)
					}
				}
			}

		}()
	}

	wg.Wait()

	// sample noise
	sampler = sample.NewUniformRange(new(big.Int).Add(new(big.Int).Neg(boundN), big.NewInt(1)), boundN)
	noise, _ := sampler.Sample()

	if debug {
		fmt.Println("starting Function Key generation")
	}

	// derive a functional key for vector c
	start = time.Now()
	key, _ := fe.DeriveKey(c, noise, masterSecKey)

	timeDK := time.Since(start)

	if debug {
		fmt.Println("time Function Key generation: ", timeDK)
	}

	// terminate code if the decryption process would be to costly
	if !coeffManipulation && numClient > 300 {
		return timeSetup, timeEnc, timeDK, 0
	}

	//performs decryption process with or without search step
	var timeDec time.Duration
	if !logSearch {

		start = time.Now()
		fe.DecryptWOSearch(cipher, key, pubKey)
		timeDec = time.Since(start)

		if debug {
			fmt.Println("time Decryption WO Search: ", timeDec)
		}
	} else {

		start = time.Now()
		_, err := fe.DecryptWOSearch(cipher, key, pubKey)
		if err != nil {
			fmt.Println(err)
		}
		timeDec = time.Since(start)

		if debug {
			fmt.Println("time Decryption WITH Search: ", timeDec)
		}

	}

	return timeSetup, timeEnc, timeDK, timeDec
}

// runs the QUILT scheme for given input parameters and outputs the needed time per algorithm
func runQUILT(secLevel, vecLen, numClient, sparseLevel int, boundX, boundY, boundN *big.Int, x data.Matrix, debug bool) (time.Duration, time.Duration, time.Duration, time.Duration) {
	if debug {
		fmt.Println("******************QUILT******************")
		fmt.Println("N:", numClient*vecLen, "numClients: ", numClient, " VecLen: ", vecLen, "sparseLevel:", sparseLevel)
	}
	label := make([]byte, 16)

	// build the scheme
	fe := NewOTNMCFE(secLevel, numClient, vecLen, boundX, boundY, boundN)

	// generate master secret key, encryption keys and public key
	start := time.Now()
	masterSecKey, enckeys, pubKey, _ := fe.GenerateKeys()
	timeSetup := time.Since(start)
	if debug {
		fmt.Println("time Setup: ", timeSetup)
	}

	//Perform Encryption
	cipher := make([]*OTNMCFECT, numClient)
	var wg sync.WaitGroup
	var timeEnc time.Duration
	maxWorkers := runtime.NumCPU()
	sem := make(chan struct{}, maxWorkers)
	startT := time.Now()

	for i := 0; i < numClient; i++ {
		i := i
		wg.Add(1)

		go func() {
			defer wg.Done()
			sem <- struct{}{}
			defer func() { <-sem }()

			if i == 0 {
				start = time.Now()
				cipher[i], _ = fe.Encrypt(enckeys[i], x[i], label)
				timeEnc = time.Since(start)
				if debug {
					fmt.Println("Time for one encryption:", timeEnc)
				}
			} else {
				// For other encryptions
				cipher[i], _ = fe.Encrypt(enckeys[i], x[i], label)

			}
		}()

	}
	wg.Wait()

	if debug {
		fmt.Println("Time encryption total:", time.Since(startT))
	}

	// sample inner product vectors and put them in a matrix
	// sparselevel describes the number of coefficients != 0
	sampler := sample.NewUniformRange(new(big.Int).Add(new(big.Int).Neg(boundY), big.NewInt(1)), boundY)
	yQuad := make([][]data.Matrix, numClient)
	yLin, _ := data.NewRandomMatrix(numClient, vecLen, sampler)
	yCon, _ := sampler.Sample()

	counter := 0

	var counterMutex sync.Mutex
	sem = make(chan struct{}, maxWorkers)

	for i := 0; i < numClient; i++ {
		i := i
		wg.Add(1)
		go func() {
			defer wg.Done()
			sem <- struct{}{}
			defer func() { <-sem }()

			yQuad[i] = make([]data.Matrix, vecLen)
			for j := 0; j < vecLen; j++ {
				yQuad[i][j] = data.NewConstantMatrix(numClient, vecLen, big.NewInt(0))
				for k := 0; k < numClient; k++ {
					for l := 0; l < vecLen; l++ {
						if i < k || (i == k && j <= l) {
							if !coeffManipulation {
								yQuad[i][j][k][l], _ = sampler.Sample()
								continue
							}
							counterMutex.Lock()
							if counter < sparseLevel {
								counter++
								counterMutex.Unlock()
								yQuad[i][j][k][l], _ = sampler.Sample()
								continue
							}
							counterMutex.Unlock()
						}
						yQuad[i][j][k][l] = big.NewInt(0)
					}
				}
			}

		}()
	}

	wg.Wait()

	// sample noise
	sampler = sample.NewUniformRange(new(big.Int).Add(new(big.Int).Neg(boundN), big.NewInt(1)), boundN)
	noise, _ := sampler.Sample()

	if debug {
		fmt.Println("starting Function Key generation")
	}

	// derive a functional key for vector c
	start = time.Now()
	key, _ := fe.DeriveKey(yQuad, yLin, yCon, noise, label, masterSecKey)
	timeDK := time.Since(start)

	if debug {
		fmt.Println("time Function Key generation: ", timeDK)
	}

	var timeDec time.Duration
	if !logSearch {
		// decryptor decrypts the quadratic function without knowing
		//vectors x and c
		start = time.Now()
		fe.DecryptWOSearch(key, yQuad, cipher, pubKey)
		timeDec = time.Since(start)

		if debug {
			fmt.Println("time Decryption WO Search: ", timeDec)
		}
	} else {

		start = time.Now()
		_, err := fe.Decrypt(key, yQuad, cipher, pubKey)
		if err != nil {
			fmt.Println(err)
		}
		timeDec = time.Since(start)

		if debug {
			fmt.Println("time Decryption WITH Search: ", timeDec)
		}

	}

	return timeSetup, timeEnc, timeDK, timeDec

}

// writes results into file
func write(filename string, message string, append bool) {

	var file *os.File
	var err error

	if append {
		file, err = os.OpenFile(filename, os.O_RDWR|os.O_CREATE|os.O_APPEND, 0755)
	} else {
		file, err = os.OpenFile(filename, os.O_RDWR|os.O_CREATE|os.O_TRUNC, 0755)
	}

	if err != nil {
		log.Fatal(err)
	}
	defer file.Close()

	_, err = file.WriteString(message)
	if err != nil {
		log.Fatal(err)
	}

}
