## Load the library
if (!requireNamespace("BiocManager", quietly = TRUE))
  install.packages("BiocManager")
BiocManager::install("flowCore")
library("flowCore")

## Some functions to help...
## Calls the density function, scales the result, and sets x-and y-lims
format_density = function(sig, xlim=F, ylim=F, ...) {
  d = density(sig, ...)
  x = d$x
  y = d$y * d$n
  
  ## Apply xlim
  xind = seq(1, length(d$x))
  if(length(xlim) > 1) {
    xind = which(d$x > xlim[1] & d$x < xlim[2])
  } 
  
  ## Apply ylim
  yind = seq(1, length(d$y))
  if(length(ylim) > 1) {
    yind = which(d$y > ylim[1] & d$y < ylim[2])
  }
  
  ind = intersect(xind, yind)
  return(data.frame("x"=x[ind], "y"=y[ind]))
}

plot_area_under_density = function(x, y, ...) {
  polygon(c(x, max(x), min(x), min(x)), c(y, 0, 0, y[1]), ...)
}

# setwd("C:\\Users\\xxliy\\Desktop\\Hartemink Lab\\FACS\\cell_cycle/")
setwd('/Users/trung/Research/deconvolution-python/data/facs/cell_cycle/')

filenames=list.files(path = './Replicate 1/',pattern = glob2rx('*.fcs'),full.names = T)
t=sub('.fcs','',filenames)
t=sub('./Replicate 1/Specimen_001_','',t)
t=as.numeric(sub('min_...','',t))

idx=order(t,decreasing = F)
filenames=filenames[idx]
t=t[idx]
tiff('C:\\Users\\xxliy\\Desktop\\Hartemink Lab\\manuscript/figures_final/rep1_FACS.tiff',width = 8000,height = 2000,res = 600,pointsize=10,compression='lzw')
par(mfrow=c(2,8),mar=c(2,2,2,2),oma=c(2,3,0,0))
fname = filenames[1]

## Read in the data...
fcs = read.FCS(fname, transformation=F)

## Name of the channel of interest
d = 'Alexa Fluor 488-A'

## If you want to change the limits of either axis (xlim or ylim),
## change either xlim or ylim in format density call... use "F" if you dont want to change anything.
d.sig = format_density(fcs@exprs[,d],bw=1,xlim=c(50,600))
plot(d.sig, type='l', xlim=c(0,600),main=expression(paste(alpha,'-factor')),lwd=2,bty='n')
plot_area_under_density(d.sig[,'x'], d.sig[,'y'], border=F, col="grey90")



for (i in 2:length(filenames)){
fname = filenames[i]

## Read in the data...
fcs = read.FCS(fname, transformation=F)

## Name of the channel of interest
d = 'Alexa Fluor 488-A'

## If you want to change the limits of either axis (xlim or ylim),
## change either xlim or ylim in format density call... use "F" if you dont want to change anything.
d.sig = format_density(fcs@exprs[,d],bw=1,xlim=c(50,600))
plot(d.sig, type='l', xlim=c(0,600),main=paste0(t[i],' min'),lwd=2,bty='n')
plot_area_under_density(d.sig[,'x'], d.sig[,'y'], border=F, col="grey90")
}
mtext('DNA content',side=1,line=1,outer = T)
mtext('Cell counts', side=2,line=1,outer = T)
dev.off()

filenames=list.files(path = './Replicate 2/',pattern = glob2rx('*.fcs'),full.names = T)
t=sub('.fcs','',filenames)
t=sub('./Replicate 2/Specimen_001_','',t)
t=as.numeric(sub('min_...','',t))

idx=order(t,decreasing = F)
filenames=filenames[idx]
t=t[idx]
tiff('C:\\Users\\xxliy\\Desktop\\Hartemink Lab\\manuscript/figures_final/rep2_FACS.tiff',width = 8000,height = 2000,res = 600,pointsize=10,compression='lzw')
par(mfrow=c(2,8),mar=c(2,2,2,2),oma=c(2,3,0,0))
fname = filenames[1]

## Read in the data...
fcs = read.FCS(fname, transformation=F)

## Name of the channel of interest
d = 'Alexa Fluor 488-A'

## If you want to change the limits of either axis (xlim or ylim),
## change either xlim or ylim in format density call... use "F" if you dont want to change anything.
d.sig = format_density(fcs@exprs[,d],bw=1,xlim=c(50,600))
plot(d.sig, type='l', xlim=c(0,600),main=expression(paste(alpha,'-factor')),lwd=2,bty='n')
plot_area_under_density(d.sig[,'x'], d.sig[,'y'], border=F, col="grey90")


for (i in 2:length(filenames)){
  fname = filenames[i]
  
  ## Read in the data...
  fcs = read.FCS(fname, transformation=F)
  
  ## Name of the channel of interest
  d = 'Alexa Fluor 488-A'
  
  ## If you want to change the limits of either axis (xlim or ylim),
  ## change either xlim or ylim in format density call... use "F" if you dont want to change anything.
  d.sig = format_density(fcs@exprs[,d],bw=1,xlim=c(50,600))
  plot(d.sig, type='l', xlim=c(0,600),main=paste0(t[i],' min'),lwd=2,bty='n')
  plot_area_under_density(d.sig[,'x'], d.sig[,'y'], border=F, col="grey90")
}
mtext('DNA content',side=1,line=1,outer = T)
mtext('Cell counts', side=2,line=1,outer = T)

dev.off()

