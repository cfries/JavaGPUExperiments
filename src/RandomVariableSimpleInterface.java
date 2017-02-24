
public interface RandomVariableSimpleInterface {

	long size();

	float[] getRealizations();

	RandomVariableSimpleInterface add(RandomVariableSimpleInterface randomVariable);

	RandomVariableSimpleInterface div(RandomVariableSimpleInterface randomVariable);

}
