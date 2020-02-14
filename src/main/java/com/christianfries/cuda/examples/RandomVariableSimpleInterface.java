/*
 * (c) Copyright Christian P. Fries, Germany. All rights reserved. Contact: email@christian-fries.de.
 *
 * Created on 24.02.2017
 */

package com.christianfries.cuda.examples;

public interface RandomVariableSimpleInterface {

	long size();

	float[] getRealizations();

	RandomVariableSimpleInterface add(RandomVariableSimpleInterface randomVariable);

	RandomVariableSimpleInterface div(RandomVariableSimpleInterface randomVariable);

}
