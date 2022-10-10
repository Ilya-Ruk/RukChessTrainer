#ifndef DATA_H
#define DATA_H

#include "types.h"

void LoadEntries(char* path, DataSet* data, int n);
void ShuffleData(DataSet* data);

#endif