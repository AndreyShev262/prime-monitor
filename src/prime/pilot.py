import math
from statistics import NormalDist

z = NormalDist().inv_cdf


def n_per_group(sigma, mde, alpha=0.05, power=0.8, comparisons=1):
    """Число клиентов на группу для заданного MDE."""
    za = z(1 - alpha / (2 * comparisons))
    zb = z(power)
    return math.ceil(2 * (za + zb) ** 2 * sigma ** 2 / mde ** 2)


def mde_for_n(sigma, n, alpha=0.05, power=0.8, comparisons=1):
    """Минимальный обнаруживаемый эффект при n клиентах на группу."""
    za = z(1 - alpha / (2 * comparisons))
    zb = z(power)
    return (za + zb) * sigma * math.sqrt(2 / n)


def diff_ci(t, c, alpha=0.05):
    """Разность средних test − control, доверительный интервал и p-значение."""
    d = t.mean() - c.mean()
    se = (t.var() / len(t) + c.var() / len(c)) ** 0.5
    q = z(1 - alpha / 2)
    p = 2 * (1 - NormalDist().cdf(abs(d / se)))
    return {"diff": d, "lo": d - q * se, "hi": d + q * se, "p": p}


def srm_pvalue(sizes):
    """p-значение проверки равных долей для двух или трёх групп."""
    expected = sum(sizes) / len(sizes)
    chi2 = sum((size - expected) ** 2 / expected for size in sizes)

    if len(sizes) == 2:
        return math.erfc(math.sqrt(chi2 / 2))
    if len(sizes) == 3:
        return math.exp(-chi2 / 2)
    raise ValueError("Для четырёх и более групп используйте scipy.stats.chisquare")


def cuped(y, x):
    """Скорректированная метрика и коэффициент theta."""
    theta = y.cov(x) / x.var()
    return y - theta * (x - x.mean()), theta