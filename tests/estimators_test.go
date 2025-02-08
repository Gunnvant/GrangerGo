package testing

import (
	"granger/estimator"
	"testing"
)

func TestAREstimatorNonNullModelData(t *testing.T) {
	lag := 2
	ts := []float64{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14}
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

func TestAREstimator2Series(t *testing.T) {
	seriesX := []float64{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
	seriesY := []float64{-11, 120, 17, 1, 0, 1, 49, 18, 19, 20}
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
	seriesX := []float64{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20}
	maxLag := estimator.GetMaxLagDefault(2)
	bestLagAIC, bestLagBIC := estimator.SelectBestLag(seriesX, maxLag)

	if bestLagAIC < 0 || bestLagAIC > 4 {
		t.Error("Lag can't be more than maxLag or less than 1")
	}
	if bestLagBIC < 0 || bestLagBIC > 4 {
		t.Error("Lag can't be more than maxLag or less than 1")
	}
}

func TestGrangerCausality(t *testing.T) {
	seriesX := []float64{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
	seriesY := []float64{-11, 120, 17, 1, 0, 1, 49, 18, 19, 20}
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
