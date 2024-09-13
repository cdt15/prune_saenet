if (!require(mice)) {
    quit("no", 2)
}

library(mice)

# command args
temp_dir <- NULL
seed <- NULL

args <- commandArgs(trailingOnly=TRUE)
for (arg in args) {
    s <- strsplit(arg, "=")[[1]]
    if (length(s) < 2) {
        next
    }

    if (s[1] == "--temp_dir") {
        temp_dir <- paste(s[2:length(s)], collapse="=")
    } else if (s[1] == "--rs") {
        seed <- strtoi(s[2])
    }
}

if (!is.null(seed))
    set.seed(seed)

# args for mice
X <- NULL
n_imputations <- NULL
maxit <- NULL
is_discrete <- NULL

# X
path <- file.path(temp_dir, "X.csv")
X <- read.csv(path, sep=',', header=FALSE)

# n_imputations
path <- file.path(temp_dir, "n_imputations.csv")
n_imputations <- read.csv(path, sep=',', header=FALSE)[1, 1]

# maxit
path <- file.path(temp_dir, "maxit.csv")
maxit <- read.csv(path, sep=',', header=FALSE)[1, 1]

# is_discrete
path <- file.path(temp_dir, "is_discrete.csv")
if (file.exists(path)) {
    is_discrete <- read.csv(path, sep=",", header=FALSE)
    is_discrete <- as.vector(is_discrete)
}

# run mice
# XXX: 一回目は、離散を含んでもmethの指定はいらないのだろうか？
mids <- mice(X, m=n_imputations, method="norm", maxit=maxit)

meth <- mids$method
if (!is.null(is_discrete)) {
    for (i in 1:length(is_discrete)) {
        if (!is_discrete[[i]]) next
        meth[names(meth)[i]] <- "logreg"
    }
}

mids <- mice(X, m=n_imputations, method=meth, maxit=maxit)
X_imputed <- lapply(1:n_imputations, function(i) complete(mids, action = i))

# output
names <- c()
for (i in 1:n_imputations) {
    fname <- paste("X_imputed_", sprintf("%08d", i), ".csv", sep="")
    path <- file.path(temp_dir, fname)
    write.csv(X_imputed[i], path, row.names=FALSE)
    
    names <- c(names, fname)
}

path <- file.path(temp_dir, "result_filenames.csv")
write.csv(names, path, row.names=F, quote=F)

quit("no", 0)
