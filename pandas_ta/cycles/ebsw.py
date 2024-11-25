# -*- coding: utf-8 -*-
from numpy import cos, exp, mean, nan, pi, roll, sin, sqrt, zeros
from pandas import DataFrame, Series
from pandas_ta._typing import DictLike, Int
from pandas_ta.utils import v_offset, v_pos_default, v_series


def ebsw(
    close: Series, length: Int = None, bars: Int = None,
    initial_version: bool = False,
    offset: Int = None, **kwargs: DictLike
) -> Series:
    """Even Better SineWave (EBSW)

    This indicator measures market cycles and uses a low pass filter to
    remove noise. Its output is bound signal between -1 and 1 and the
    maximum length of a detected trend is limited by its length input.

    Coded by rengel8 for Pandas TA based on a publication at
    'prorealcode.com' and a book by J.F.Ehlers. According to the suggestion
    by Squigglez2* and major differences between the initial version's
    output close to the implementation from Ehler's, the default version is
    now more closely related to the code from pro-realcode.

    Remark:
    The default version is now more cycle oriented and tends to be less
    whipsaw-prune. Thus the older version might offer earlier signals at
    medium and stronger reversals. A test against the version at TradingView
    showed very close results with the advantage to be one bar/candle faster,
    than the corresponding reference value. This might be pre-roll related
    and was not further investigated.
    * https://github.com/twopirllc/pandas-ta/issues/350

    Sources:
        - https://www.prorealcode.com/prorealtime-indicators/even-better-sinewave/
        - J.F.Ehlers 'Cycle Analytics for Traders', 2014

    Args:
        close (pd.Series): Series of 'close's
        length (int): It's max cycle/trend period. Values between 40-48
            work like expected with minimum value: 39. Default: 40.
        bars (int): Period of low pass filtering. Default: 10
        drift (int): The difference period. Default: 1
        offset (int): How many periods to offset the result. Default: 0

    Kwargs:
        fillna (value, optional): pd.DataFrame.fillna(value)
        fill_method (value, optional): Type of fill method

    Returns:
        pd.Series: New feature generated.
    """
    # Validate
    length = v_pos_default(length, 40)
    close = v_series(close, length)

    if close is None:
        return

    bars = v_pos_default(bars, 10)
    offset = v_offset(offset)

    # Calculate
    # allow initial version to be used (more responsive/caution!)
    m = close.size

    if isinstance(initial_version, bool) and initial_version:
        # not the default version that is active
        alpha1 = hp = 0  # alpha and HighPass
        a1 = b1 = c1 = c2 = c3 = 0
        filter_ = power_ = wave = 0
        lastClose = lastHP = 0
        filtHist = [0, 0]   # Filter history

        result = [nan for _ in range(0, length - 1)] + [0]
        for i in range(length, m):
            # HighPass filter cyclic components whose periods are shorter than
            # Duration input
            alpha1 = (1 - sin(360 / length)) / cos(360 / length)
            hp = 0.5 * (1 + alpha1) * (close[i] - lastClose) + alpha1 * lastHP

            # Smooth with a Super Smoother Filter from equation 3-3
            a1 = exp(-sqrt(2) * pi / bars)
            b1 = 2 * a1 * cos(sqrt(2) * 180 / bars)
            c2 = b1
            c3 = -1 * a1 * a1
            c1 = 1 - c2 - c3
            filter_ = 0.5 * c1 * (hp + lastHP) + c2 * \
                filtHist[1] + c3 * filtHist[0]
            # filter_ = float("{:.8f}".format(float(filter_))) # to fix for
            # small scientific notations, the big ones fail

            # 3 Bar average of wave amplitude and power
            wave = (filter_ + filtHist[1] + filtHist[0]) / 3
            power_ = filter_ * filter_ + filtHist[1] * filtHist[1] \
                + filtHist[0] * filtHist[0]
            power_ /= 3
            # Normalize the Average Wave to Square Root of the Average Power
            wave = wave / sqrt(power_)

            # update storage, result
            filtHist.append(filter_)  # append new filter_ value
            # remove first element of list (left) -> updating/trim
            filtHist.pop(0)
            lastHP = hp
            lastClose = close[i]
            result.append(wave)

    else:  # this version is the default version
        # Calculate
        lastHP = lastClose = 0
        filtHist = zeros(3)
        result = [nan] * (length - 1) + [0]

        angle = 2 * pi / length
        alpha1 = (1 - sin(angle)) / cos(angle)
        ang = 2 ** .5 * pi / bars
        a1 = exp(-ang)
        c2 = 2 * a1 * cos(ang)
        c3 = -a1 ** 2
        c1 = 1 - c2 - c3

        for i in range(length, m):
            hp = 0.5 * (1 + alpha1) * (close[i] - lastClose) + alpha1 * lastHP

            # Rotate filters to overwrite oldest value
            filtHist = roll(filtHist, -1)
            filtHist[-1] = 0.5 * c1 * \
                (hp + lastHP) + c2 * filtHist[1] + c3 * filtHist[0]

            # Wave calculation
            wave = mean(filtHist)
            rms = sqrt(mean(filtHist ** 2))
            wave = wave / rms

            # Update past values
            lastHP = hp
            lastClose = close[i]
            result.append(wave)

    ebsw = Series(result, index=close.index)

    # Offset
    if offset != 0:
        ebsw = ebsw.shift(offset)

    # Fill
    if "fillna" in kwargs:
        ebsw.fillna(kwargs["fillna"], inplace=True)
    if "fill_method" in kwargs:
        ebsw.fillna(method=kwargs["fill_method"], inplace=True)

    # Name and Category
    ebsw.name = f"EBSW_{length}_{bars}"
    ebsw.category = "cycles"

    return ebsw

def ebsw_extra(
        close: Series, length: Int = None, 
        lbR: Int = None, lbL: Int = None, 
        range_upper: Int = None, range_lower: Int = None
) -> Series:
    """Even Better SineWave (EBSW) Extra
    
    This indicator measures market cycles and uses a low pass filter to
    remove noise. Its output is bound signal between -1 and 1 and the
    maximum length of a detected trend is limited by its length input.
    
    This version of EBSW includes divergence signals for bullish, bearish,
    hidden bullish, and hidden bearish divergences.
    
    Sources:
        - https://www.prorealcode.com/prorealtime-indicators/even-better-sinewave/
        - J.F.Ehlers 'Cycle Analytics for Traders', 2014
        - BlackCat's divergence calculations on TradingView
        
    Args:
        close (pd.Series): Series of 'close's
        length (int): It's max cycle/trend period. Values between 40-48 
        work like expected with minimum value: 39. Default: 40.
        lbL (int): Left lookback period for divergence calculations. Default: 5
        lbR (int): Right lookback period for divergence calculations. Default: 5
        range_upper (int): Upper range for divergence calculations. Default: 60
        range_lower (int): Lower range for divergence calculations. Default: 5
    
    Returns:
        pd.Series: New feature generated.
    """

    # Validate
    length = v_pos_default(length, 40)
    lbL = v_pos_default(lbL, 5)
    lbR = v_pos_default(lbR, 5)
    range_upper = v_pos_default(range_upper, 60)
    range_lower = v_pos_default(range_lower, 5)
    close = v_series(close, length)

    if close is None:
        return
    
    # Initialize variables
    hp = zeros(len(close))
    filt = zeros(len(close))
    
    # Calculate alpha1
    alpha1 = (1 - sin(2 * pi / length)) / cos(2 * pi / length)
    
    # High-pass filter
    for i in range(1, len(close)):
        hp[i] = 0.5 * (1 + alpha1) * (close[i] - close[i-1]) + alpha1 * hp[i-1]
    
    # Super Smoother Filter coefficients
    a1 = exp(-1.414 * pi / 10)
    b1 = 2 * a1 * cos(1.414 * pi / 10)
    c2 = b1
    c3 = -a1 * a1
    c1 = 1 - c2 - c3
    
    # Apply Super Smoother Filter
    for i in range(2, len(close)):
        filt[i] = c1 * (hp[i] + hp[i-1]) / 2 + c2 * filt[i-1] + c3 * filt[i-2]
    
    # Calculate Wave and Power
    wave = Series(filt).rolling(window=3).mean()
    power = Series(filt**2).rolling(window=3).mean()
    
    # Normalize Wave
    wave = wave / sqrt(power)
    
    # Calculate Trigger (previous Wave value)
    trigger = wave.shift(1)
    
    # Divergence calculations
    def find_pivots(data, left, right):
        pivots = zeros(len(data))
        for i in range(left, len(data) - right):
            if all(data[i] > data[i-left:i]) and all(data[i] > data[i+1:i+right+1]):
                pivots[i] = 1  # High
            elif all(data[i] < data[i-left:i]) and all(data[i] < data[i+1:i+right+1]):
                pivots[i] = -1  # Low
        return pivots

    pivots = find_pivots(wave.values, lbL, lbR)
    
    def calculate_divergence(price, osc, pivots):
        bullish = zeros(len(price))
        bearish = zeros(len(price))
        hidden_bullish = zeros(len(price))
        hidden_bearish = zeros(len(price))
        
        for i in range(range_upper, len(price)):
            if pivots[i] == -1:  # Low pivot
                for j in range(i - range_upper, i - range_lower):
                    if pivots[j] == -1:
                        if price[i] < price[j] and osc[i] > osc[j]:
                            bullish[i] = 1
                        elif price[i] > price[j] and osc[i] < osc[j]:
                            hidden_bullish[i] = 1
                        break
            elif pivots[i] == 1:  # High pivot
                for j in range(i - range_upper, i - range_lower):
                    if pivots[j] == 1:
                        if price[i] > price[j] and osc[i] < osc[j]:
                            bearish[i] = 1
                        elif price[i] < price[j] and osc[i] > osc[j]:
                            hidden_bearish[i] = 1
                        break
        
        return bullish, bearish, hidden_bullish, hidden_bearish

    bullish, bearish, hidden_bullish, hidden_bearish = calculate_divergence(close.values, wave.values, pivots)
    
    return DataFrame({
        'Wave': wave,
        'Trigger': trigger,
        'Bullish': bullish,
        'Bearish': bearish,
        'Hidden_Bullish': hidden_bullish,
        'Hidden_Bearish': hidden_bearish
    })