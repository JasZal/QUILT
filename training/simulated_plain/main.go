package main

import (
	"encoding/csv"
	"fmt"
	"log"
	"math"
	"os"
	"strconv"

	"github.com/google/differential-privacy/go/noise"
)

var noisy bool

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

func UNUSED(t ...interface{}) {}

func main() {
	iterations := 25

	noisy = true

	rounds := 10

	//alpha := 0.01
	//eps := 9.0
	prefix := "../datasets/training"
	files := []string{"LBW", "UIS", "PCS"}
	postfix := ".csv"
	fileR := "results.txt"

	var epsilon []float64
	for i := 0.00; i < 10.0; i += 0.2 {
		if i == 0.0 {
			epsilon = append(epsilon, 0.01)
		} else {
			epsilon = append(epsilon, i)
		}
	}
	//epsilon = []float64{5.0}

	scalingStart := 10000
	scalingEnd := 1000
	scalingStepSize := 9000
	alpha := []float64{0.01, 0.03, 0.06, 0.09, 0.1, 0.3, 0.6, 0.9}

	if noisy {
		write(fileR, fmt.Sprintf("epsilon: %v\n", epsilon))
	}

	for _, training := range files {
		fmt.Printf("training: %v\n", training)

		for scaling := scalingStart; scaling >= scalingEnd; scaling -= scalingStepSize {

			test := prefix + training + postfix
			data, attr := loadData(prefix+training+postfix, float64(scaling))
			testdata, _ := loadData(test, math.NaN())
			//fmt.Printf("data: %v\n", data)

			theta0 := make([]float64, attr+1)
			del := 1.0 / float64(len(data))
			batch := len(data)
			fmt.Printf("batch: %v\n", batch)

			fmt.Printf(" scaling: %v ", scaling)
			write(fileR, fmt.Sprintf(training+"%v = [", scaling))
			for _, e := range epsilon {
				fmt.Printf("e= %v\n", e)
				max := make([]float64, 2)
				for _, a := range alpha {
					acc := 0.0
					for r := 0; r < rounds; r++ {

						theta := gradientDescent(data, iterations, a, float64(scaling), e, del, theta0)

						UNUSED(theta0, del, iterations)

						//fmt.Println(compAcc(testdata, theta))
						acc += compAcc(testdata, theta)

					}
					if acc/float64(rounds) >= max[1] {

						max[0] = a
						max[1] = acc / float64(rounds)
					}

					//log Reg
					fmt.Printf("acc average: %v\n", acc/float64(rounds))

				}
				fmt.Printf("-- max: %v\n", max)
				write(fileR, fmt.Sprintf("%v, ", max[1]))

			}
			write(fileR, "];\n")
			fmt.Println("*************")

		}
	}

}

func loadData(file string, scaling float64) ([][]float64, int) {
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

	attr := len(records[0])
	//fmt.Printf("attr: %v\n", attr)

	data := make([][]float64, len(records))
	for i := 0; i < len(records); i++ {
		data[i] = make([]float64, len(records[0])+1)
		if math.IsNaN(scaling) {
			data[i][0] = 1
		} else {
			data[i][0] = 1 * scaling
		}

		for j := 0; j < attr; j++ {
			interm, _ := strconv.ParseFloat(records[i][j], 64)
			if !math.IsNaN(scaling) {
				data[i][j+1] = math.Round(interm * scaling)
			} else {
				data[i][j+1] = interm
			}
		}
	}

	//	fmt.Printf("data: %v\n", data)
	return data, attr - 1
}

func h(x, theta []float64) float64 {
	sum := 0.0
	for i := 0; i < len(x); i++ {
		sum += x[i] * (theta[i])
	}

	return 0.5 + 0.25*sum

}

func computeInfSen(theta []float64, alpha, b float64) float64 {
	sum := 0.0
	for i := 0; i < len(theta); i++ {
		sum += math.Abs(theta[i])
	}

	return alpha / b * (1 + 0.25*sum)

}

func compute0Sen(theta []float64) int64 {
	return int64(len(theta))
}

func computeNoise(theta []float64, alpha, b, scaling, eps, del float64) []float64 {
	n := make([]float64, len(theta))

	infSen := computeInfSen(theta, alpha, b)

	zeroSen := compute0Sen(theta)

	//noise via gauss

	gauss := noise.Gaussian()
	for j := 0; j < len(n); j++ {
		n[j] = gauss.AddNoiseFloat64(0.0, zeroSen, infSen, eps, del)

	}
	//fmt.Printf("n: %v\n", n)

	//scale noise

	for j := 0; j < len(n); j++ {

		n[j] *= math.Pow(scaling, 3)

		if n[j] >= 0 {
			n[j] = math.Ceil(n[j])
		} else {
			n[j] = math.Floor(n[j])
		}

		n[j] = n[j] / math.Pow(scaling, 3)

	}

	if !noisy {
		n = make([]float64, len(theta))
	}
	return n
}

func gradientDescentIteration(data [][]float64, alpha, scaling, eps, delta float64, theta []float64) []float64 {
	b := float64(len(data))
	//fmt.Printf("theta: %v\n", theta)
	noise := computeNoise(theta, alpha, b, scaling, eps, delta)
	//fmt.Printf("noise: %v\n", noise)
	thetaN := make([]float64, len(theta))

	//bias
	thetaN[0] = math.Round((theta[0] - alpha/2 - (alpha*theta[0])/(4)) * math.Pow(scaling, 3))
	for i := 0; i < len(data); i++ {
		sum := 0.0
		for k := 1; k < len(data[i])-1; k++ {
			sum += math.Round((-1*alpha*theta[k]/(4*b))*math.Pow(scaling, 2)) * data[i][k]
		}

		thetaN[0] += math.Round((alpha/b)*math.Pow(scaling, 2))*data[i][len(data[i])-1] + sum

	}
	thetaN[0] = thetaN[0] / math.Pow(float64(scaling), 3)
	//	thetaN[0] += noise[0]

	//other weights
	for j := 1; j < len(theta); j++ {
		thetaN[j] = math.Round((theta[j]) * math.Pow(scaling, 3))

		for i := 0; i < len(data); i++ {
			sum := 0.0
			for k := 1; k < len(data[i])-1; k++ {
				sum += math.Round(((-1*alpha*theta[k])/(4*b))*scaling) * data[i][k] * data[i][j]
			}

			thetaN[j] += math.Round((alpha/b)*scaling)*data[i][len(data[i])-1]*data[i][j] + sum + math.Round(((-1*alpha*(2+theta[0]))/(4*b))*math.Pow(scaling, 2))*data[i][j] //+ math.Round(noise[j]*math.Pow(scaling, 3))
		}
		thetaN[j] = thetaN[j] / math.Pow(float64(scaling), 3)
		//thetaN[j] += noise[j]

	}

	// fmt.Printf("thetaN: %v\n", thetaN)
	// fmt.Printf("noise: %v\n", noise)
	for i := 0; i < len(theta); i++ {
		thetaN[i] += noise[i]
	}

	//fmt.Printf("thetaN: %v\n", thetaN)

	return thetaN
}

func gradientDescent(data [][]float64, it int, alpha, scaling, eps, del float64, theta []float64) []float64 {

	e := 0.0

	for i := 0; i < it; i++ {

		eI := eps*math.Pow(float64(i+1)/float64(it), 1.5) - eps*math.Pow(float64(i)/float64(it), 1.5)
		e += eI

		theta = gradientDescentIteration(data, alpha, scaling, eI, del/float64(it), theta)

		//fmt.Printf("Iteration %v: theta= %v\n", i, theta)

	}

	return theta

}

func compAcc(data [][]float64, theta []float64) float64 {

	sum := 0.0
	for i := 0; i < len(data); i++ {
		x := 0.0
		if h(data[i][0:len(data[0])-1], theta) >= 0.5 {
			x = 1.0
		}

		if x == data[i][len(data[0])-1] {
			sum += 1
		}

	}
	return sum / float64(len(data))

}
