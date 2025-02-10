package testing

import (
	"granger/estimator"
	"testing"

	"gonum.org/v1/gonum/mat"
)

func TestCreateXYLag1(t *testing.T) {
	ts := []float64{1, 2, 3, 4, 5, 6}
	lag := 1
	ExpectedY := mat.NewDense(5, 1, []float64{2, 3, 4, 5, 6})
	ExpectedX := mat.NewDense(5, 2, []float64{
		1, 1,
		1, 2,
		1, 3,
		1, 4,
		1, 5})
	res := estimator.CreateXY(ts, lag)
	if !mat.Equal(res.Y, ExpectedY) {
		t.Errorf("Y is: %v while expected was: %v", res.Y, ExpectedY)
	}
	if !mat.Equal(res.X, ExpectedX) {
		t.Errorf("X is: %v while expected was: %v", res.X, ExpectedX)
	}
}

func TestCreateXYLag2(t *testing.T) {
	ts := []float64{1, 2, 3, 4, 5, 6}
	lag := 2
	ExpectedY := mat.NewDense(4, 1, []float64{3, 4, 5, 6})
	ExpectedX := mat.NewDense(4, 3, []float64{
		1, 1, 2,
		1, 2, 3,
		1, 3, 4,
		1, 4, 5})
	res := estimator.CreateXY(ts, lag)
	if !mat.Equal(res.Y, ExpectedY) {
		t.Errorf("Y is: %v while expected was: %v", res.Y, ExpectedY)
	}
	if !mat.Equal(res.X, ExpectedX) {
		t.Errorf("X is: %v while expected was: %v", res.X, ExpectedX)
	}
}

func TestCreateXYLag3(t *testing.T) {
	ts := []float64{1, 2, 3, 4, 5, 6}
	lag := 3
	ExpectedY := mat.NewDense(3, 1, []float64{4, 5, 6})
	ExpectedX := mat.NewDense(3, 4, []float64{1, 1, 2, 3,
		1, 2, 3, 4,
		1, 3, 4, 5})
	res := estimator.CreateXY(ts, lag)
	if !mat.Equal(res.Y, ExpectedY) {
		t.Errorf("Y is: %v while expected was: %v", res.Y, ExpectedY)
	}
	if !mat.Equal(res.X, ExpectedX) {
		t.Errorf("X is: %v while expected was: %v", res.X, ExpectedX)
	}
}

func TestCreateXXyLag1(t *testing.T) {
	seriesY := []float64{1, 2, 3, 4, 5, 6}
	seriesX := []float64{7, 8, 9, 10, 11, 12}
	lag := 1
	ExpectedY := mat.NewDense(5, 1, []float64{2, 3, 4, 5, 6})
	ExpectedX := mat.NewDense(5, 3, []float64{
		1, 1, 7,
		1, 2, 8,
		1, 3, 9,
		1, 4, 10,
		1, 5, 11})
	res := estimator.CreateXXy(seriesY, seriesX, lag)
	if !mat.Equal(res.Y, ExpectedY) {
		t.Errorf("Y is: %v while expected was: %v", res.Y, ExpectedY)
	}
	if !mat.Equal(res.X, ExpectedX) {
		t.Errorf("X is: %v while expected was: %v", res.X, ExpectedX)
	}

}
func TestCreateDataLag2(t *testing.T) {
	seriesY := []float64{1, 2, 3, 4, 5, 6}
	seriesX := []float64{7, 8, 9, 10, 11, 12}
	lag := 2
	ExpectedY := mat.NewDense(4, 1, []float64{3, 4, 5, 6})
	ExpectedX := mat.NewDense(4, 5, []float64{
		1, 1, 2, 7, 8,
		1, 2, 3, 8, 9,
		1, 3, 4, 9, 10,
		1, 4, 5, 10, 11})
	res := estimator.CreateXXy(seriesY, seriesX, lag)
	if !mat.Equal(res.Y, ExpectedY) {
		t.Errorf("Y is: %v while expected was: %v", res.Y, ExpectedY)
	}
	if !mat.Equal(res.X, ExpectedX) {
		t.Errorf("X is: %v while expected was: %v", res.X, ExpectedX)
	}
}

func TestComputeFStat(t *testing.T) {
	rssRestricted := 10.0
	rssUnrestricted := 15.0
	lag := 2
	n := 100
	fstat := estimator.ComputeFstat(&rssRestricted, &rssUnrestricted, lag, n)
	if fstat < 0 {
		t.Error("F Statistic can't be negative")
	}
}
