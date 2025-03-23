package estimator

import (
	"math"
	"sync"

	"gonum.org/v1/gonum/mat"
)

func CreateXY(ts []float64, lag int) *DataU {
	n := len(ts)
	nrows := n - lag
	ncols := lag
	X := mat.NewDense(nrows, ncols, nil)
	Y := mat.NewDense(nrows, 1, nil)
	Ones := mat.NewDense(nrows, 1, make([]float64, nrows))
	for i := 0; i < nrows; i++ {
		Ones.Set(i, 0, 1.0)
	}
	for i := lag; i < n; i++ {
		Y.Set(i-lag, 0, ts[i])
	}
	for i := 0; i < nrows; i++ {
		for j := 0; j < ncols; j++ {
			X.Set(i, j, ts[i+j])
		}
	}
	rows1, cols1 := Ones.Dims()
	_, cols2 := X.Dims()
	combined := mat.NewDense(nrows, cols1+cols2, nil)
	combined.Slice(0, rows1, 0, cols1).(*mat.Dense).Copy(Ones)
	combined.Slice(0, rows1, cols1, cols1+cols2).(*mat.Dense).Copy(X)
	res := &DataU{}
	res.Lag = lag
	res.X = combined
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
	Ones := mat.NewDense(nrows, 1, make([]float64, nrows))
	for i := 0; i < nrows; i++ {
		Ones.Set(i, 0, 1.0)
	}
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
	X3 := mat.NewDense(rows1, cols1+cols2, nil)
	X3.Slice(0, rows1, 0, cols1).(*mat.Dense).Copy(X1)
	X3.Slice(0, rows1, cols1, cols1+cols2).(*mat.Dense).Copy(X2)

	rowsOne, colsOne := Ones.Dims()
	_, cols3 := X3.Dims()
	combined := mat.NewDense(rowsOne, colsOne+cols3, nil)
	combined.Slice(0, rowsOne, 0, colsOne).(*mat.Dense).Copy(Ones)
	combined.Slice(0, rowsOne, colsOne, colsOne+cols3).(*mat.Dense).Copy(X3)

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
func GetAicBic(ts []float64, chRes chan AicBic, wg *sync.WaitGroup, lag int) {
	res := CreateXY(ts, lag)
	regRes := ARfit(res)
	chRes <- AicBic{AIC: *regRes.AIC, BIC: *regRes.BIC, AicLag: lag, BicLag: lag}
	wg.Done()

}
func SelectBestLagParallel(ts []float64, maxLag int) (int, int) {
	bestAIC, bestBIC := math.Inf(1), math.Inf(1)
	bestLagAIC, bestLagBIC := 1, 1
	chRes := make(chan AicBic, maxLag)
	wg := sync.WaitGroup{}
	for lag := 1; lag <= maxLag; lag++ {
		wg.Add(1)
		go GetAicBic(ts, chRes, &wg, lag)
	}
	wg.Wait()
	close(chRes)
	for res := range chRes {
		if res.AIC < bestAIC {
			bestAIC = res.AIC
			bestLagAIC = res.AicLag
		}
		if res.BIC < bestBIC {
			bestBIC = res.BIC
			bestLagBIC = res.BicLag
		}
	}

	return bestLagAIC, bestLagBIC
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

func ComputeFstat(rssRestricted *float64, rssUnrestricted *float64, lag int, n int) float64 {
	df_unrestricted := n - (2*lag + 1)
	df_restricted := lag
	result := ((*rssRestricted - *rssUnrestricted) / float64(df_restricted)) / (*rssUnrestricted / float64(df_unrestricted))
	if result < 0 {
		result = result * -1.0
		return result
	} else {
		return result
	}

}
func GrangerCausality(seriesY []float64, seriesX []float64, maxLag int) *GrangerResult {
	bestLagAICY, bestLagBICY := SelectBestLag(seriesY, maxLag)
	bestLagAICX, bestLagBICX := SelectBestLag(seriesX, maxLag)

	// Get Models for Y on X AIC Criteria
	resAICYX := CreateXXy(seriesY, seriesX, bestLagAICX)
	resAICY := CreateXY(seriesY, bestLagAICX)
	rssRYXAIC := ARfit(resAICY).RSS
	rssUYXAIC := ARfit(resAICYX).RSS
	FStatYXAIC := ComputeFstat(rssRYXAIC, rssUYXAIC, bestLagAICX, len(seriesX))

	//Get Models Y on X BIC Criteria
	resBICYX := CreateXXy(seriesY, seriesX, bestLagBICX)
	resBICY := CreateXY(seriesY, bestLagBICX)
	rssRYXBIC := ARfit(resBICY).RSS
	rssUYXBIC := ARfit(resBICYX).RSS
	FStatYXBIC := ComputeFstat(rssRYXBIC, rssUYXBIC, bestLagBICX, len(seriesX))

	//Get Models X on Y AIC Criteria
	resAICXY := CreateXXy(seriesX, seriesY, bestLagAICY)
	resAICX := CreateXY(seriesX, bestLagAICY)
	rssRXYAIC := ARfit(resAICX).RSS
	rssUXYAIC := ARfit(resAICXY).RSS
	FStatXYAIC := ComputeFstat(rssUXYAIC, rssRXYAIC, bestLagAICY, len(seriesY))

	//Get Models X on Y BIC Criteria
	resBICXY := CreateXXy(seriesX, seriesY, bestLagBICY)
	resBICX := CreateXY(seriesX, bestLagBICY)
	rssRXYBIC := ARfit(resBICX).RSS
	rssUXYBIC := ARfit(resBICXY).RSS
	FStatXYBIC := ComputeFstat(rssUXYBIC, rssRXYBIC, bestLagBICY, len(seriesX))

	result := GrangerResult{}
	result.FStat_YX_AIC = FStatYXAIC
	result.FStat_YX_BIC = FStatYXBIC
	result.FStat_XY_AIC = FStatXYAIC
	result.FStat_XY_BIC = FStatXYBIC
	return &result

}

func GrangerCausalityParallel(seriesY []float64, seriesX []float64, maxLag int) *GrangerResult {
	bestLagAICY, bestLagBICY := SelectBestLagParallel(seriesY, maxLag)
	bestLagAICX, bestLagBICX := SelectBestLagParallel(seriesX, maxLag)

	// Get Models for Y on X AIC Criteria
	resAICYX := CreateXXy(seriesY, seriesX, bestLagAICX)
	resAICY := CreateXY(seriesY, bestLagAICX)
	rssRYXAIC := ARfit(resAICY).RSS
	rssUYXAIC := ARfit(resAICYX).RSS
	FStatYXAIC := ComputeFstat(rssRYXAIC, rssUYXAIC, bestLagAICX, len(seriesX))

	//Get Models Y on X BIC Criteria
	resBICYX := CreateXXy(seriesY, seriesX, bestLagBICX)
	resBICY := CreateXY(seriesY, bestLagBICX)
	rssRYXBIC := ARfit(resBICY).RSS
	rssUYXBIC := ARfit(resBICYX).RSS
	FStatYXBIC := ComputeFstat(rssRYXBIC, rssUYXBIC, bestLagBICX, len(seriesX))

	//Get Models X on Y AIC Criteria
	resAICXY := CreateXXy(seriesX, seriesY, bestLagAICY)
	resAICX := CreateXY(seriesX, bestLagAICY)
	rssRXYAIC := ARfit(resAICX).RSS
	rssUXYAIC := ARfit(resAICXY).RSS
	FStatXYAIC := ComputeFstat(rssUXYAIC, rssRXYAIC, bestLagAICY, len(seriesY))

	//Get Models X on Y BIC Criteria
	resBICXY := CreateXXy(seriesX, seriesY, bestLagBICY)
	resBICX := CreateXY(seriesX, bestLagBICY)
	rssRXYBIC := ARfit(resBICX).RSS
	rssUXYBIC := ARfit(resBICXY).RSS
	FStatXYBIC := ComputeFstat(rssUXYBIC, rssRXYBIC, bestLagBICY, len(seriesX))

	result := GrangerResult{}
	result.FStat_YX_AIC = FStatYXAIC
	result.FStat_YX_BIC = FStatYXBIC
	result.FStat_XY_AIC = FStatXYAIC
	result.FStat_XY_BIC = FStatXYBIC
	return &result
}
