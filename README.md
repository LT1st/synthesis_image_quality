# synthesis_image_quality
To get the comparation of datapairs(raw and fake). SSIM, PSNR, MSSIM, PID, FID, and many other method are included.

### Documentation for Evaluation Dataset Class

#### Introduction
This Python script provides a comprehensive framework for evaluating super-resolution (SR) image generation models, particularly focusing on generating and assessing thermal images. It integrates several key image quality metrics including Peak Signal-to-Noise Ratio (PSNR), Structural Similarity Index (SSIM), Learned Perceptual Image Patch Similarity (LPIPS), and Perceptual Image Distance (PID) using a pre-trained VGG network. Additionally, the script supports generating datasets for Frechet Inception Distance (FID) score calculations, a popular metric for evaluating the quality of generated images compared to real images.

#### Usage
1. **Setting up the Environment:**
   - Ensure Python 3.x is installed.
   - Install the required libraries: PyTorch, torchvision, OpenCV, scikit-image, LPiPS, NumPy, Pandas, Matplotlib, tqdm, and the pytorch_fid package.
   - Clone or download the script to your local environment.

2. **Initializing the Evaluation Dataset:**
   - Create an instance of the `EvalDataSet` class. You must specify the path to the folder containing the images to be evaluated (e.g., SR generated images and corresponding high-resolution images). Optionally, specify the device ('cuda' for GPU or 'cpu' for CPU), dataset name, and dataset type.
     ```python
     folder_path = "path_to_your_images"
     eval_dataset = EvalDataSet(folder_path=folder_path)
     ```

3. **Generating FID Datasets:**
   - Call the `gen_FID_dataset()` method to generate datasets for FID evaluation. You can specify the root path for the new dataset; the method organizes generated and real images into separate folders within this root directory.
     ```python
     eval_dataset.gen_FID_dataset(target_new_dataset_root="./FID_data")
     ```

4. **Calculating FID Score:**
   - Use the `FID_score()` method to compute the FID score between your generated images and the original high-resolution images. You can specify the batch size, dimensions for feature extraction, and path to the dataset.
     ```python
     fid_value = eval_dataset.FID_score(bs=4, dims=2048, path="./FID_data")
     print(fid_value)
     ```

5. **Processing Images and Calculating Metrics:**
   - To calculate PSNR, SSIM, LPIPS, and PID for each pair of generated and original images, use the `process_imgs()` method. This function processes all pairs and stores the results internally within the class instance.
     ```python
     ssim, psnr, lpips, pid = eval_dataset.process_imgs()
     ```

6. **Data Visualization and Export:**
   - To visualize the distribution and average of a metric, use `visualize_list_and_average()`. For example, to visualize SSIM values:
     ```python
     eval_dataset.visualize_list_and_average(eval_dataset.ssim)
     ```
   - Convert metric lists into a pandas DataFrame and save them to a CSV file using `get_df()`:
     ```python
     df = eval_dataset.get_df("output_metrics.csv")
     ```
   - To load metric values from a CSV file into the dataset instance, use `load_df()`:
     ```python
     df = eval_dataset.load_df("path_to_your_csv")
     ```

7. **Analyzing Results:**
   - After storing metrics in a DataFrame, you can easily calculate mean and variance for each metric:
     ```python
     mean_values = df.mean()
     variance_values = df.var()
     print(mean_values, variance_values)
     ```

#### Notes
- Make sure that the images for evaluation (both generated and original) are correctly named and matched in the specified folder. The default expected naming convention is that the SR images end with 'sr.png' and the corresponding HR images with 'hr.png'.
- Ensure that your computing environment has sufficient memory and processing power, as image processing and model evaluation can be resource-intensive.
- The script assumes the images are in RGB format. If your images are in a different format, you might need to adjust the image loading and processing steps accordingly.
