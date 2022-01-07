Official Tensorflow implementation of papers: 

*Explaining the Black-box Smoothly- A Counterfactual Approach*  [**Paper**](https://arxiv.org/abs/2101.04230)

*Using Causal Analysis for Conceptual Deep Learning Explanation* [**Paper**](https://arxiv.org/abs/2107.06098)

# Usage

## Train baseline black-box classification model
1. Download StanfordCheXpert and MIMIC-CXR dataset

2. Train a classifier. Skip this step if you have a pretrained classifier. 
```
python Step_1_train_classifier.py --config 'configs/Step_1_StanfordCheXpert_Classifier_256.yaml'
```

3. Save the output of the trained classifier. User can choose to save the predictions on test set of the same dataset or some different dataset. Set the config appropriately. We used this file to save the prediction on MIMIC-CXR dataset.

```
python Step_1_test_classifier.py --config 'configs/Step_1_StanfordCheXpert_Classifier_256.yaml'
```

## Generate counterfactual visual explanation
4. Process the output of the classifier and create input for Explanation model by discretizing the posterior probability. We used this file to create training dataset for Explainer for Pleural Effusion, Cardiomegaly and Edema.

```
./notebooks/Step_1_Test_Classifier_&_Create_Input_For_Explainer.ipynb
```
5. Train a Segmentation Network. We train a lung segmentation network on [**JSRT**](http://db.jsrt.or.jp/eng.php) dataset

```
python Step_2_train_segmentation.py --config 'configs/Step_2_JSRT_Segmentation_256.yaml'
```

6. Train a Object Detector for pacemaker and hardware. The code is borrowed from: [Faster_RCNN_TensorFlow](https://github.com/MingtaoGuo/Faster_RCNN_TensorFlow.git)
```
python Step_3_train_fast_RCNN.py --config 'configs/Step_3_MIMIC_OD_Pacemaker.yaml'
```

7. Trainer explainer model.
```
python Step_4_train_explainer.py --config 'configs/Step_4_MIMIC_Explainer_256_Pleural_Effusion.yaml'
```

8. Explore the trained Explanation model and see qualitative results.
```
python Step_4_test_explainer.py --config 'configs/Step_4_MIMIC_Explainer_256_Pleural_Effusion.yaml'
./notebooks/Quantitative_Results.ipynb
```

## Generate conceptual explanation
9. Dissect the trained classification model to identify hidden units which are relevant for a given concept
```
./notebooks/Step_5_Find_Concept_Units_From_MAX_Activations.ipynb
python Step_1_test_classifier.py --config 'configs/Step_1_StanfordCheXpert_Classifier_256_temp.yaml'
Rscript --vanilla Step_5_Lasso_Regression.r

```

10. Use this notebook to visualize the activation regions for a given hidden unit
```
./notebooks/Step_6_Visualize_Hidden_Units.ipynb

```

11. Use the counterfactual explanation to compute the causal in-direct effect associated with a given set of concept-units

```
./notebooks/Step_7_Causal_Indirect_Effect.ipynb
python Step_4_test_explainer.py --config 'configs/Step_4_MIMIC_Explainer_256_Pleural_Effusion.yaml'

```


# Cite

```
@article{
ExplainingBBSmoothly,
  author    = {Sumedha Singla and
               Brian Pollack and
               Stephen Wallace and
               Kayhan Batmanghelich},
  title     = {Explaining the Black-box Smoothly- {A} Counterfactual Approach},
  volume    = {abs/2101.04230},
  year      = {2021},
  url       = {https://arxiv.org/abs/2101.04230}
}
```

```
@article {
Singla2021Explanation,
	Title = {Using Causal Analysis for Conceptual Deep Learning Explanation},
	Author = {Singla, Sumedha and Wallace, Stephen and Triantafillou, Sofia and Batmanghelich, Kayhan},
	DOI = {10.1007/978-3-030-87199-4_49},
	Volume = {12903},
	Year = {2021},
	Journal = {Medical image computing and computer-assisted intervention : MICCAI ... International Conference on Medical Image Computing and Computer-Assisted Intervention},
	Pages = {519—528}
}
```






