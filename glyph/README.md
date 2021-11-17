Generate glyph values using TensorFlow through Google Colabs.

Copy predict_glyph_handlabel into a Colabs notebook and download the HDF5 format saved model to predict glyph values from steerable pyramid features.

To generate the 100-dimensional features that the model uses follow glyph_feature_extraction.py. It imports an image, imports the transform that the model was trained on, generates the SP features for the image, and transforms them into the model feature set. From there they are saved and can be used as input features for the model included in the data folder to generate a glyph value. 
