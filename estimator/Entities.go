package estimator

import (
	"gonum.org/v1/gonum/mat"
)

type Model interface {
	Fit()
}

type ARModel struct {
	AIC   *float64
	BIC   *float64
	Coeff *mat.Dense
	RSS   *float64
}

type GrangerResult struct {
	FStat_XY_AIC float64
	FStat_XY_BIC float64
	FStat_YX_AIC float64
	FStat_YX_BIC float64
}

type DataU struct {
	Lag int
	N   int
	X   *mat.Dense
	Y   *mat.Dense
}
