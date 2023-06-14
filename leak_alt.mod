: passive leak current

NEURON {
	SUFFIX leak
	NONSPECIFIC_CURRENT il
	RANGE il, el, glbar
}

UNITS {
	(mA) = (milliamp)
	(mV) = (millivolt)
}

ASSIGNED {
	v (mV)
	il (mA/cm2)
}

BREAKPOINT { 
	il = glbar * (v - el)
}	