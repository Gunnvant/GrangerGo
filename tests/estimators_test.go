package testing

import (
	"granger/estimator"
	"testing"
)

func TestAREstimatorNonNullModelData(t *testing.T) {
	lag := 2
	ts := []float64{0.5, -0.14, 0.65, 1.52, -0.23, -0.23, 1.58, 0.77, -0.47, 0.54, -0.46, -0.47, 0.24, -1.91}
	res := estimator.CreateXY(ts, lag)
	regRes := estimator.ARfit(res)
	if regRes.AIC == nil {
		t.Error("AIC can't be nil")
	}
	if regRes.BIC == nil {
		t.Error("BIC can't be nil")
	}
	if regRes.RSS == nil {
		t.Error("RSS can't be nil")
	}
	zero := 0.0
	if *regRes.RSS < zero {
		t.Error("RSS can't be negative")

	}

}

func TestAREstimatorRSSNonNegative(t *testing.T) {
	lag := 5
	ts := []float64{0.5, -0.14, 0.65, 1.52, -0.23, -0.23, 1.58, 0.77, -0.47, 0.54, -0.46, -0.47, 0.24, -1.91}
	res := estimator.CreateXY(ts, lag)
	regRes := estimator.ARfit(res)
	if *regRes.RSS < 0 {
		t.Error("RSS can't be negative")
	}

}
func TestAREstimator2Series(t *testing.T) {
	seriesX := []float64{0.5, -0.14, 0.65, 1.52, -0.23, -0.23, 1.58, 0.77, -0.47, 0.54, -0.46, -0.47, 0.24, -1.91}
	seriesY := []float64{0.55, -0.25, -0.33, 1.57, -2.96, 0.05, -0.22, -1.43, 2.17, 2.18, 1.72, -0.55, 1.3, -2.23}
	lag := 2
	res := estimator.CreateXXy(seriesY, seriesX, lag)
	regRes := estimator.ARfit(res)
	if regRes.AIC == nil {
		t.Error("AIC can't be nil")
	}
	if regRes.BIC == nil {
		t.Error("BIC can't be nil")
	}
	if regRes.RSS == nil {
		t.Error("RSS can't be nil")
	}
	zero := 0.0
	if *regRes.RSS < zero {
		t.Error("RSS can't be negative")

	}

}

func TestBestLagSelection(t *testing.T) {
	seriesX := []float64{0.5, -0.14, 0.65, 1.52, -0.23, -0.23, 1.58, 0.77, -0.47, 0.54, -0.46, -0.47, 0.24, -1.91}
	maxLag := estimator.GetMaxLagDefault(4)
	bestLagAIC, bestLagBIC := estimator.SelectBestLag(seriesX, maxLag)

	if bestLagAIC < 0 || bestLagAIC > 4 {
		t.Error("Lag can't be more than maxLag or less than 1")
	}
	if bestLagBIC < 0 || bestLagBIC > 4 {
		t.Error("Lag can't be more than maxLag or less than 1")
	}
}

func TestGrangerCausality(t *testing.T) {
	seriesX := []float64{0.5, -0.14, 0.65, 1.52, -0.23, -0.23, 1.58, 0.77, -0.47, 0.54, -0.46, -0.47, 0.24, -1.91}
	seriesY := []float64{0.55, -0.25, -0.33, 1.57, -2.96, 0.05, -0.22, -1.43, 2.17, 2.18, 1.72, -0.55, 1.3, -2.23}
	lag := 2
	result := estimator.GrangerCausality(seriesY, seriesX, lag)
	if result.FStat_XY_AIC < 0 {
		t.Error("Fstat can't be less than zero ")

	}
	if result.FStat_XY_BIC < 0 {
		t.Error("Fstat can't be less than zero ")

	}
	if result.FStat_YX_AIC < 0 {
		t.Error("Fstat can't be less than zero ")

	}

	if result.FStat_YX_BIC < 0 {
		t.Error("Fstat can't be less than zero ")

	}
}

func TestGrangerCausalityParallel(t *testing.T) {
	seriesX := []float64{0.5, -0.14, 0.65, 1.52, -0.23, -0.23, 1.58, 0.77, -0.47, 0.54, -0.46, -0.47, 0.24, -1.91}
	seriesY := []float64{0.55, -0.25, -0.33, 1.57, -2.96, 0.05, -0.22, -1.43, 2.17, 2.18, 1.72, -0.55, 1.3, -2.23}
	lag := 2
	result := estimator.GrangerCausalityParallel(seriesY, seriesX, lag)
	if result.FStat_XY_AIC < 0 {
		t.Error("Fstat can't be less than zero ")

	}
	if result.FStat_XY_BIC < 0 {
		t.Error("Fstat can't be less than zero ")

	}
	if result.FStat_YX_AIC < 0 {
		t.Error("Fstat can't be less than zero ")

	}

	if result.FStat_YX_BIC < 0 {
		t.Error("Fstat can't be less than zero ")

	}
}
