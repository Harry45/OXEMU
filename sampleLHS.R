library(lhs)
n = 500
d = 5
X = maximinLHS(n, d)
write.csv(X, paste('data/lhs_', toString(n), '.csv', sep=''))
