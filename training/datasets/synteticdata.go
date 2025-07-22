package main

import (
	"encoding/csv"
	"fmt"
	"log"
	"math"
	"math/rand/v2"
	"os"
	"strconv"
)

func main() {
	attr := 100
	rec := 50
	theta := make([]float64, attr)
	filename := fmt.Sprintf("syntetic%vx%v.csv", rec, attr)

	for i := 0; i < attr; i++ {
		theta[i] = rand.Float64()*2 - 1
	}
	fmt.Printf("theta: %v\n", theta)

	fakeDataSet(theta, rec, filename)
}

func computeLabelLog(data [][]string, theta []float64) [][]string {

	for i := 0; i < len(data); i++ {

		sum := 0.0

		for j := 0; j < len(data[0])-1; j++ {
			interm, _ := strconv.ParseFloat(data[i][j], 64)
			sum += interm * theta[j]
		}

		y := 1.0 / (1.0 + math.Exp(-1.0*sum))

		if y < 0.5 {
			data[i][len(data[0])-1] = "0"
		} else {
			data[i][len(data[0])-1] = "1"
		}
	}

	return data
}

func fakeDataSet(theta []float64, n int, file string) {
	records := make([][]string, n)
	for i := 0; i < n; i++ {
		records[i] = make([]string, len(theta))
		for j := 0; j < len(theta); j++ {
			records[i][j] = fmt.Sprintf("%v", rand.Float64())
		}
	}

	// fake label according to theta
	computeLabelLog(records, theta)

	sets := map[string][][]string{
		file: records,
	}

	for fn, dataset := range sets {
		f, err := os.Create(fn)
		if err != nil {
			log.Fatal(err)
		}
		defer f.Close()
		out := csv.NewWriter(f)
		if err := out.WriteAll(dataset); err != nil {
			log.Fatal(err)
		}
		out.Flush()
	}
}
