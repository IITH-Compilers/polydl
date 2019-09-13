//On Data Split of 70% Training and 30% Testing
//Accuracy - 61.4% with equality between 1%
if(v2-L1DataSetSize*lower_bound <= v1-L1DataSetSize <= v2-L1DataSetSize*upper_bound){
// Middle
	if(v1-L3DataSetSize*lower_bound <= v2-L3DataSetSize <= v1-L3DataSetSize*upper_bound){
	//Middle
	Both Version are equal ==> 1
	}
	
	else if (v2-L3DataSetSize < v1-L3DataSetSize){
	//Left
	Version 2 is better ==> 0
	}

	else if (v2-L3DataSetSize < v1-L3DataSetSize)
	{
	//Right
	Version 1 is better ==> 2		
	}
}

else if(v1-L1DataSetSize < v2-L1DataSetSize){
//Left
	if(v1-PessiL1DataSetSize*lower_bound <= v2-PessiL1DataSetSize <= v1-PessiL1DataSetSize*upper_bound){
	//Middle
	Version 2 is better ==> 0
	}
	
	else if (v2-PessiL1DataSetSize < v1-PessiL1DataSetSize){
	//Left
	Version 1 is better ==> 2
	}

	else if (v2-PessiL1DataSetSize < /* > */ v1-PessiL1DataSetSize)
	{
	//Right
	Version 1 is better ==> 2		
	}

}

else if(v1-L1DataSetSize > v2-L1DataSetSize)
{
//Right
	if(v2-MemDataSetSize*lower_bound <= v1-MemDataSetSize <= v2-MemDataSetSize*upper_bound){
	//Middle
	Version 2 is better ==> 0
	}
	
	else if (v1-MemDataSetSize < v2-MemDataSetSize){
	//Left
	Version 2 is better ==> 0
	}

	else if (v1-MemDataSetSize < v2-MemDataSetSize)
	{
	//Right
	Version 2 is better ==> 0		
	}
}
