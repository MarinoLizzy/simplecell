TITLE potassium for simple cell

UNITS {
	(mA) = (milliamp)
	(mV) = (millivolt)
}

PARAMETER {
    gkbar = 0.009 (siemens/cm2)
}

ASSIGNED {
    v (mV)
    ninf  
    ntau (ms)
    ek (mV) 
    ik (mA/cm2)
}

NEURON {
    SUFFIX k
    USEION k READ ek WRITE ik
    RANGE gkbar
    RANGE ninf, ntau
}

INITIAL {
    rates(v)
    n = ninf
}

STATE {
    n
}

BREAKPOINT {
    SOLVE states METHOD cnexp
    ik = gkbar * n * n * n * n * (v - ek)
}

DERIVATIVE states {
    rates(v)
    n' = (ninf - n) / ntau
}

PROCEDURE rates(v(mV)) {
    LOCAL alphan, betan
    TABLE ninf, ntau FROM -100 TO 100 WITH 200
    
    if (v != -34) {
        alphan = (-0.01 * (v + 34)) / (exp(-0.1 * (v + 34)) - 1)
    } else {
        alphan = (-0.01 * (-33.9 + 34)) / (exp(-0.1 * (-33.9 + 34)) - 1)
    }
    betan = 0.125 * exp(-(v + 44) / 80)

    ninf = alphan / (alphan + betan)
    ntau = 1 / (alphan + betan)
}