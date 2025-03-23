package main

import (
	"encoding/csv"
	"fmt"
	"granger/estimator"
	"log"
	"os"
	"runtime"
	"runtime/pprof"
	"strconv"
	"time"
)

func readCsvFile(filePath string) []map[string]string {
	f, err := os.Open(filePath)
	if err != nil {
		log.Fatal("Unable to read input file "+filePath, err)
	}
	defer f.Close()

	csvReader := csv.NewReader(f)
	records, err := csvReader.ReadAll()
	if err != nil {
		log.Fatal("Unable to parse file as CSV for "+filePath, err)
	}
	headers := records[0]
	data := []map[string]string{}
	for _, row := range records[1:] {
		rowMap := make(map[string]string)
		for i, value := range row {
			rowMap[headers[i]] = value
		}
		data = append(data, rowMap)
	}
	return data
}
func main() {
	start := time.Now()
	var memStart runtime.MemStats
	runtime.ReadMemStats(&memStart)
	f, err := os.Create("cpu.prof")
	if err != nil {
		log.Fatal(err)
	}
	defer f.Close()

	pprof.StartCPUProfile(f)
	defer pprof.StopCPUProfile()
	records := readCsvFile("./datasets/benchmark_large.csv")
	seriesX := make([]float64, 0, len(records))
	seriesY := make([]float64, 0, len(records))
	for _, val := range records {
		x, _ := strconv.ParseFloat(val["x"], 64)
		y1, _ := strconv.ParseFloat(val["y_1"], 64)
		seriesX = append(seriesX, x)
		seriesY = append(seriesY, y1)
	}
	for i := 0; i < 5; i++ {
		lag := 5
		result := estimator.GrangerCausalityParallel(seriesY, seriesX, lag)
		fmt.Printf("Results:  %v", *result)
	}
	var memEnd runtime.MemStats
	runtime.ReadMemStats(&memEnd)
	elapsed := time.Since(start)
	fmt.Printf("Time taken: %s\n", elapsed)
	fmt.Printf("Memory allocated: %d bytes\n", memEnd.Alloc-memStart.Alloc)
}
