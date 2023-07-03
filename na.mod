TITLE Mod file for Na in simple cell

UNITS {
	(mA) = (milliamp)
	(mV) = (millivolt)
}

NEURON {
    SUFFIX na
    USEION na READ ena WRITE ina
    RANGE gnabar
    RANGE minf, hinf, mtau, htau
}

INITIAL {
    rates(v)
    m = minf
    h = hinf
}

STATE {
    m h
}

PARAMETER {
    gnabar = 0.035 (siemens/cm2)
}

ASSIGNED {
    v (mV)
    minf 
    mtau (ms) 
    hinf 
    htau (ms)
    ena  (mV)
    ina  (mA/cm2)
}

BREAKPOINT {
    SOLVE states METHOD cnexp
    ina = gnabar * m * m * m * h * (v - ena)
}

DERIVATIVE states {
    rates(v)
    m' = (minf - m) / mtau
    h' = (hinf - h) / htau
}

PROCEDURE rates(v(mV)) {
    LOCAL alpham, betam, alphah, betah
    TABLE minf, hinf, mtau, htau FROM -100 TO 100 WITH 200

    if (v != -35) {
        alpham = (-0.1 * (v + 35)) / (exp(-0.1 * (v + 35)) - 1)
    } else {
        alpham = (-0.1 * (-34.9 + 35)) / (exp(-0.1 * (-34.9 + 35)) - 1)
    }
    betam = 4 * exp(-(v + 60) / 18)
    alphah = 0.07 * exp(-(v + 58) / 20)
    betah = 1 / (exp(-0.1 * (v + 28)) + 1)

    minf = alpham / (alpham + betam)
    mtau = 1 / (alpham + betam)
    hinf = alphah / (alphah + betah)
    htau = 1 / (alphah + betah)
}
