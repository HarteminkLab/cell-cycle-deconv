function R2 = rsquare(y, yhat)

yhat= reshape(yhat,1,size(yhat,1)*size(yhat,2)); 
y= reshape(y,1,size(y,1)*size(y,2));
R2 = corr2(y, yhat)^2;

return;
