TITLE potassium for simple cell

COMMENT
I just took the info from Na and converted it to K -- need to check that this is correct
ENDCOMMENT

UNITS {
	(mA) = (milliamp)
	(mV) = (millivolt)
}

NEURON {
    SUFFIX k
    USEION k READ ek WRITE ik
    RANGE gkbar
    RANGE ninf, ntau
}

STATE {
    n
}

BREAKPOINT {
    SOLVE states METHOD cnexp
    ik = gkbar * (v - ek)
}

DERIVATIVE states {
    rates(v)
    n' = (ninf - n) / ntau
}

PROCEDURE rates(v(mV)) {
    LOCAL alphan, betan
    TABLE ninf, ntau FROM -100 TO 100 WITH 200

    alphan = (-0.01 (v + 34)) / (exp(-0.1 (v + 34)) - 1)
    betan = 0.125 * exp(-(v + 44) / 80)

    ninf = alphan / (alphan + betan)
    ntau = 1 / (alphan + betan)
}