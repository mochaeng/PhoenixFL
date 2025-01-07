#ifndef __CLASSIFIER_H__
#define __CLASSIFIER_H__

#ifdef __cplusplus

extern "C" {
#endif

typedef void *mModel;

mModel NewModel(const char *modelFile);
int PredictIsPositiveBinary(mModel model, float *inputData, int numFeatures);
void DeleteModel(mModel model);

#ifdef __cplusplus
}
#endif

#endif
