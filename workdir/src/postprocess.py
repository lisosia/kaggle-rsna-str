def calib_p(arr, factor):  # set factor>1 to enhance positive prob
    return arr * factor / (arr * factor + (1-arr))

