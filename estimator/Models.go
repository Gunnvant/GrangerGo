package estimator

import (
	"math"

	"gonum.org/v1/gonum/mat"
)

func CreateXY(ts []float64, lag int) *DataU {
	n := len(ts)
	nrows := n - lag
	ncols := lag
	X := mat.NewDense(nrows, ncols, nil)
	Y := mat.NewDense(nrows, 1, nil)
	for i := lag; i < n; i++ {
		Y.Set(i-lag, 0, ts[i])
	}
	for i := 0; i < nrows; i++ {
		for j := 0; j < ncols; j++ {
			X.Set(i, j, ts[i+j])
		}
	}
	res := &DataU{}
	res.Lag = lag
	res.X = X
	res.Y = Y
	res.N = n
	return res
}
func CreateXXy(seriesY []float64, seriesX []float64, lag int) *DataU {
	n := len(seriesY)
	nrows := n - lag
	X1 := mat.NewDense(nrows, lag, nil)
	X2 := mat.NewDense(nrows, lag, nil)
	Y := mat.NewDense(nrows, 1, nil)
	for i := lag; i < n; i++ {
		Y.Set(i-lag, 0, seriesY[i])
	}
	for i := 0; i < nrows; i++ {
		for j := 0; j < lag; j++ {
			X1.Set(i, j, seriesY[i+j])
		}
	}
	for i := 0; i < nrows; i++ {
		for j := 0; j < lag; j++ {
			X2.Set(i, j, seriesX[i+j])
		}
	}
	rows1, cols1 := X1.Dims()
	_, cols2 := X2.Dims()
	combined := mat.NewDense(rows1, cols1+cols2, nil)
	combined.Slice(0, rows1, 0, cols1).(*mat.Dense).Copy(X1)
	combined.Slice(0, rows1, cols1, cols1+cols2).(*mat.Dense).Copy(X2)
	res := &DataU{}
	res.Lag = lag
	res.X = combined
	res.Y = Y
	res.N = n
	return res
}

func ARfit(d *DataU) *ARModel {
	n := d.N
	X := d.X
	Y := d.Y
	// Attempt QR decomposition first
	var qr mat.QR
	qr.Factorize(X)
	beta := mat.NewDense(X.RawMatrix().Cols, 1, nil)
	err := qr.SolveTo(beta, false, Y)
	if err != nil {
		panic("Data is not well formed, can't estimate parameters")
	}
	var YHat mat.Dense
	YHat.Mul(X, beta)
	var residuals mat.Dense
	residuals.Sub(Y, &YHat)
	var rss mat.Dense
	rss.Mul(residuals.T(), &residuals)
	r := rss.At(0, 0)
	p, _ := beta.Dims()
	aic := float64(n)*math.Log(r/float64(n)) + 2*float64(p)
	bic := float64(n)*math.Log(r/float64(n)) + float64(p)*math.Log(float64(n))
	a := &ARModel{}
	a.Coeff = beta
	a.RSS = &r
	a.AIC = &aic
	a.BIC = &bic
	return a
}
func GetMaxLagDefault(maxLag int) int {
	return maxLag
}

func SelectBestLag(ts []float64, maxLag int) (int, int) {
	bestAIC, bestBIC := math.Inf(1), math.Inf(1)
	bestLagAIC, bestLagBIC := 1, 1
	for lag := 1; lag <= maxLag; lag++ {
		res := CreateXY(ts, lag)
		regRes := ARfit(res)
		if *regRes.AIC < bestAIC {
			bestAIC = *regRes.AIC
			bestLagAIC = lag
		}
		if *regRes.BIC <= bestBIC {
			bestBIC = *regRes.BIC
			bestLagBIC = lag
		}

	}
	return bestLagAIC, bestLagBIC
}

func ComputeFtat(rssRestricted *float64, rssUnrestricted *float64, lag int, n int) float64 {
	df_unrestricted := n - (2*lag + 1)
	df_restricted := lag
	result := ((*rssUnrestricted - *rssRestricted) / float64(df_restricted)) / (*rssUnrestricted / float64(df_unrestricted))
	return result

}
func GrangerCausality(seriesY []float64, seriesX []float64, maxLag int) *GrangerResult {
	bestLagAICY, bestLagBICY := SelectBestLag(seriesY, maxLag)
	bestLagAICX, bestLagBICX := SelectBestLag(seriesX, maxLag)

	// Get Models for Y on X AIC Criteria
	resAICYX := CreateXXy(seriesY, seriesX, bestLagAICX)
	resAICY := CreateXY(seriesY, bestLagAICX)
	rssUYXAIC := ARfit(resAICY).RSS
	rssRYXAIC := ARfit(resAICYX).RSS
	FStatYXAIC := ComputeFtat(rssRYXAIC, rssUYXAIC, bestLagAICX, len(seriesX))

	//Get Models Y on X BIC Criteria
	resBICYX := CreateXXy(seriesY, seriesX, bestLagBICX)
	resBICY := CreateXY(seriesY, bestLagBICX)
	rssUYXBIC := ARfit(resBICY).RSS
	rssRYXBIC := ARfit(resBICYX).RSS
	FStatYXBIC := ComputeFtat(rssRYXBIC, rssUYXBIC, bestLagBICX, len(seriesX))

	//Get Models X on Y AIC Criteria
	resAICXY := CreateXXy(seriesX, seriesY, bestLagAICY)
	resAICX := CreateXY(seriesX, bestLagAICY)
	rssUXYAIC := ARfit(resAICX).RSS
	rssRXYAIC := ARfit(resAICXY).RSS
	FStatXYAIC := ComputeFtat(rssUXYAIC, rssRXYAIC, bestLagAICY, len(seriesY))

	//Get Models X on Y BIC Criteria
	resBICXY := CreateXXy(seriesX, seriesY, bestLagBICY)
	resBICX := CreateXY(seriesX, bestLagBICY)
	rssUXYBIC := ARfit(resBICX).RSS
	rssRXYBIC := ARfit(resBICXY).RSS
	FStatXYBIC := ComputeFtat(rssUXYBIC, rssRXYBIC, bestLagBICY, len(seriesX))

	result := GrangerResult{}
	result.FStat_YX_AIC = FStatYXAIC
	result.FStat_YX_BIC = FStatYXBIC
	result.FStat_XY_AIC = FStatXYAIC
	result.FStat_XY_BIC = FStatXYBIC
	return &result

}
