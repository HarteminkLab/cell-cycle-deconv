function ratio = p2t(array)

    % To avoid dividing by zero, we insert a value called 
    % SMALL, default value is 1.
    SMALL = 1;

    r = 0.2;
    low = max(quantile(array, r), SMALL);
    high = quantile(array, 1-r);
    ratio = high/low;

return;
