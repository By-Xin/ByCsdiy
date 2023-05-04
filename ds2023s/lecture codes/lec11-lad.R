library(quantreg)
set.seed(123)
x = rnorm(20)
y = x + rnorm(20)

#windows()
plot(x, y, xlim = c(-3, 3), ylim = c(-3, 3))
reg = lm(y ~ x)
abline(reg, col = "red")
qreg = rq(y ~ x)
abline(qreg, col = "blue")
legend("bottomright", legend = c("OLS", "LAD"),
       lwd = 2, col = c("red", "blue"))

for(i in 1:10)
{
    pt = locator(n = 1, type = "p")
    x = c(x, pt$x)
    y = c(y, pt$y)
    
    # plot(x, y, xlim = c(-3, 3), ylim = c(-3, 3))
    reg = lm(y ~ x)
    abline(reg, col = "red")
    qreg = rq(y ~ x)
    abline(qreg, col = "blue")
    # legend("bottomright", legend = c("OLS", "LAD"),
    #    lwd = 2, col = c("red", "blue"))
}

