module LocalMethods

using GSL
using LinearAlgebra
using Statistics
import JLD2

export cov_n1_dot_n2, cov_angle

const data_dir = "../data/"
const cls = JLD2.FileIO.load(data_dir*"cls.jld2", "cls")
const w_on_Pl = cls[:cl_len_scalar][:,1] .* (2 .* cls[:ell] .+ 1) ./ (4π) ./ cls[:factor_on_cl_cmb]
const lmax = maximum(cls[:ell])

function cov_cosθ(cosθ,lmax::Int)
	@assert lmax >= 2 "lmax too small, must be greater than 1"
	dot(sf_legendre_Pl_array(lmax, cosθ)[3:end], w_on_Pl[3:lmax+1])
end

cov_n1_dot_n2(n1_dot_n2) = cov_cosθ(n1_dot_n2,lmax)
cov_angle(angle) = cov_cosθ(cos(angle),lmax)

function bin_mean(fk; bin=0)
	fkrtn = copy(fk)
	if bin > 1
		Nm1    = length(fk)
		subNm1 = bin * (Nm1÷bin)
		fmk    = reshape(fk[1:subNm1], bin, Nm1÷bin)
		fmk   .= mean(fmk, dims=1)
		fkrtn[1:subNm1] .= vec(fmk)
		fkrtn[subNm1+1:end] .= mean(fk[subNm1+1:end])
	end
	return fkrtn
end

end # module
