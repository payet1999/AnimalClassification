package com.animalclassification

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Matrix
import android.os.Bundle
import android.util.Log
import android.view.View
import android.widget.*
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.*
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import com.animalclassification.ml.AnimalClassificationModelOptimized
import com.animalclassification.ml.DogBreedClassificationModel
import org.tensorflow.lite.DataType
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.*
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

class MainActivity : AppCompatActivity() {

    private var imageCapture: ImageCapture? = null
    private var imageToPredict: Bitmap? = null

    private var predictedAnimalIndex: Int? = null
    private var predictedAnimalProbability: Float = 0.0f

    private var predictedDogIndex: Int? = null
    private var predictedDogProbability: Float = 0.0f
    private val dogBreedsLabels by lazy { getDogBreedsList() }

    private lateinit var cameraExecutor: ExecutorService

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        initViews()

        // Hide the action bar
        supportActionBar?.hide()

        // Check camera permissions. If all permission granted, start camera.
        // Else, ask for the permission
        if (allPermissionsGranted()) startCamera() else {
            ActivityCompat.requestPermissions(this, REQUIRED_PERMISSIONS, REQUEST_CODE_PERMISSIONS)
        }

        cameraExecutor = Executors.newSingleThreadExecutor()
    }

    override fun onBackPressed() {
        // Custom back button action
        onBackButtonPressed()
    }

    override fun onDestroy() {
        super.onDestroy()
        // Shut down the camera
        cameraExecutor.shutdown()
    }

    /**
     * Checks the camera permission
     */
    override fun onRequestPermissionsResult(
        requestCode: Int, permissions: Array<String>, grantResults:
        IntArray
    ) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode == REQUEST_CODE_PERMISSIONS) {
            // If all permissions granted , then start Camera
            if (allPermissionsGranted()) startCamera() else {
                // If permissions are not granted, present a toast to notify the user that the
                // permissions were not granted.
                runOnUiThread {
                    Toast.makeText(this, "Permissions not granted by the user.", Toast.LENGTH_SHORT)
                        .show()
                }
                finish()
            }
        }
    }

    /**
     * Initializes buttons' listeners and displays required views
     */
    private fun initViews() {
        // Hides yet unused image view created to display the taken picture
        findViewById<ImageView>(R.id.background_imageview).visibility = View.GONE
        // Set on click listener for the button of capture photo
        findViewById<ImageButton>(R.id.camera_button).setOnClickListener {
            takePhoto()
        }

        // Hides back button and adds listener to it
        findViewById<Button>(R.id.back_button).visibility = View.GONE
        findViewById<Button>(R.id.back_button).setOnClickListener {
            onBackButtonPressed()
        }
    }

    /**
     * Gets de txt file containing the dog breeds names and saves it as an array
     */
    private fun getDogBreedsList() : MutableList<String> {
        val scanner = Scanner(resources.openRawResource(R.raw.dog_breeds_labels))
        val list = mutableListOf<String>()
        scanner.use { s ->
            while (s.hasNext()) {
                list += s.next().replace("_", " ").replaceFirstChar { it.uppercase() }
            }
        }
        return list
    }

    /**
     * Resets app to init state (Buttons, views and labels)
     */
    private fun onBackButtonPressed() {
        runOnUiThread {
            // Blocks the back button and hides the taken image. Resets label
            findViewById<Button>(R.id.back_button).visibility = View.GONE
            findViewById<ImageView>(R.id.background_imageview).visibility = View.GONE
            findViewById<TextView>(R.id.prediction_textview).text =
                applicationContext.resources.getString(R.string.prediction_label)

            // Shows the take picture button
            findViewById<ImageButton>(R.id.camera_button).visibility = View.VISIBLE
        }
    }

    /**
     * Prepares camera to take pictures
     */
    private fun startCamera() {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)

        cameraProviderFuture.addListener({

            // Used to bind the lifecycle of cameras to the lifecycle owner
            val cameraProvider: ProcessCameraProvider = cameraProviderFuture.get()

            // Preview
            val preview = Preview.Builder()
                .build()
                .also {
                    it.setSurfaceProvider(
                        findViewById<PreviewView>(R.id.viewFinder).surfaceProvider)
                }

            imageCapture = ImageCapture.Builder().build()

            // Select back camera as a default
            val cameraSelector = CameraSelector.DEFAULT_BACK_CAMERA

            try {
                // Unbind use cases before rebinding
                cameraProvider.unbindAll()

                // Bind use cases to camera
                cameraProvider.bindToLifecycle(
                    this, cameraSelector, preview, imageCapture
                )

            } catch (exc: Exception) {
                Log.e(CAMERA_TAG, "Use case binding failed", exc)
            }

        }, ContextCompat.getMainExecutor(this))
    }

    /**
     * Take the photo
     */
    private fun takePhoto() {
        // Blocks the take photo button
        findViewById<ImageButton>(R.id.camera_button).visibility = View.GONE

        // Get a stable reference of the modifiable image capture use case
        val imageCapture = imageCapture ?: return
        imageCapture.targetRotation
        imageCapture.takePicture(cameraExecutor,
            object :
                ImageCapture.OnImageCapturedCallback() {
                override fun onCaptureSuccess(image: ImageProxy) {
                    super.onCaptureSuccess(image)

                    // Saves image to be displayed and analyzed as a bitmap
                    imageToPredict = imageProxyToBitmap(image)
                    runOnUiThread {
                        // Displays captured image
                        imageToPredict?.let {
                            findViewById<ImageView>(R.id.background_imageview).setImageBitmap(it)
                            findViewById<ImageView>(R.id.background_imageview).visibility = View.VISIBLE
                        }
                    }

                    // Process the image with the NN and deletes the taken image from cache
                    predictAnimal()
                    image.close()
                }

                override fun onError(exception: ImageCaptureException) {
                    super.onError(exception)

                    Log.e(CAMERA_TAG, "Photo capture failed: ${exception.message}", exception)
                }
        })
    }

    private fun allPermissionsGranted() = REQUIRED_PERMISSIONS.all {
        ContextCompat.checkSelfPermission(baseContext, it) == PackageManager.PERMISSION_GRANTED
    }

    /**
     *  Convert image proxy to bitmap
     *  @param image
     */
    private fun imageProxyToBitmap(image: ImageProxy): Bitmap {
        val planeProxy = image.planes[0]
        val buffer: ByteBuffer = planeProxy.buffer
        val bytes = ByteArray(buffer.remaining())
        buffer.get(bytes)
        val tmpBitmap = BitmapFactory.decodeByteArray(bytes, 0, bytes.size)

        return Bitmap.createScaledBitmap(
            tmpBitmap,
            (this.window.decorView.height * 0.6).toInt(),
            (this.window.decorView.width * 0.98).toInt(),
            true
        ).rotate(90f)
    }

    /**
     * Takes the saved bitmap and pas it to the ML NN model. Interprets output
     */
    private fun predictAnimal() {
        imageToPredict?.let { rawBitmap ->
            val bitmap = Bitmap.createScaledBitmap(rawBitmap, 256, 256, true)
            val model = AnimalClassificationModelOptimized.newInstance(applicationContext)

            // Creates inputs for reference.
            val inputFeature0 = TensorBuffer.createFixedSize(intArrayOf(1, 256, 256, 3), DataType.FLOAT32)

            Log.d("Animal10 model. Required shape", TensorImage.fromBitmap(bitmap).buffer.toString())
            Log.d("Animal10 model. Given shape", inputFeature0.buffer.toString())

            inputFeature0.loadBuffer(convertBitmapToByteBuffer(bitmap))

            // Runs model inference and gets result.
            val outputs = model.process(inputFeature0)
            val outputFeature0 = outputs.outputFeature0AsTensorBuffer

            // Update displayed output data (prediction and % of accuracy)
            val maxVal = outputFeature0.floatArray.indices.maxByOrNull { outputFeature0.floatArray[it] }
            Log.d("Animal10 model. Array output", maxVal.toString())
            for (value in outputFeature0.floatArray) {
                Log.d("Animal10 model. Output values", value.toString())
            }
            predictedAnimalIndex = maxVal
            maxVal?.let { predictedAnimalProbability = outputFeature0.floatArray[it] }
            // Check if predicted animal is a dog. If so, predict breed
            if (predictedAnimalIndex == DOG_INDEX) {
                predictDog(bitmap)
            } else {
                // Update UI without dog breed
                predictedDogIndex = null
                runOnUiThread {
                    setPrediction()
                }
            }

            // Releases model resources if no longer used.
            model.close()
        }
    }

    /**
     * Predicts dog breed if the animal is a dog
     */
    private fun predictDog(image: Bitmap) {
        val model = DogBreedClassificationModel.newInstance(applicationContext)

        // Creates inputs for reference.
        val inputFeature0 = TensorBuffer.createFixedSize(intArrayOf(1, 256, 256, 3), DataType.FLOAT32)
        inputFeature0.loadBuffer(convertBitmapToByteBuffer(image))

        Log.d("Dog breed model. Required shape", TensorImage.fromBitmap(image).buffer.toString())
        Log.d("Dog breed model. Given shape", inputFeature0.buffer.toString())

        // Runs model inference and gets result.
        val outputs = model.process(inputFeature0)
        val outputFeature0 = outputs.outputFeature0AsTensorBuffer

        // Update displayed output data (prediction and % of accuracy)
        val maxVal = outputFeature0.floatArray.indices.maxByOrNull { outputFeature0.floatArray[it] }
        Log.d("Dog breed model. Array output", maxVal.toString())
        predictedDogIndex = maxVal
        maxVal?.let { predictedDogProbability = outputFeature0.floatArray[it] }

        runOnUiThread {
            setPrediction()
        }

        // Releases model resources if no longer used.
        model.close()
    }

    /**
     * Updates the prediction label with the output result
     */
    private fun setPrediction() {
        var prediction = ""
        // Get bitmap from image
        predictedAnimalIndex?.let { index ->
            prediction = "Prediction:\n ${ANIMAL_LABELS[index].replaceFirstChar { it.uppercase() }} (${"%.2f".format(predictedAnimalProbability*100)}%)"
        }
        predictedDogIndex?.let { index ->
            prediction += "\n ${dogBreedsLabels[index].replaceFirstChar { it.uppercase() }} (${"%.2f".format(predictedDogProbability*100)}%)"
        }
        findViewById<TextView>(R.id.prediction_textview).text = prediction
        // Display back button
        findViewById<Button>(R.id.back_button).visibility = View.VISIBLE
    }

    /**
     * Converts Bitmap to the required input data as ByteBuffer. 256x256 image and normalized values
     */
    private fun convertBitmapToByteBuffer(bitmap: Bitmap): ByteBuffer {
        val byteBuffer =
            ByteBuffer.allocateDirect(4 * 1 * 256 * 256 * 3)
        byteBuffer.order(ByteOrder.nativeOrder())
        val intValues = IntArray(256 * 256)
        bitmap.getPixels(intValues, 0, bitmap.width, 0, 0, bitmap.width, bitmap.height)
        var pixel = 0
        for (i in 0 until 256) {
            for (j in 0 until 256) {
                val `val` = intValues[pixel++]
                byteBuffer.putFloat(((`val` shr 16 and 0xFF) - 128) / 128.0f)
                byteBuffer.putFloat(((`val` shr 8 and 0xFF) - 128) / 128.0f)
                byteBuffer.putFloat(((`val` and 0xFF) - 128) / 128.0f)
            }
        }
        return byteBuffer
    }

    companion object {
        private const val CAMERA_TAG = "CameraXGFG"
        private const val REQUEST_CODE_PERMISSIONS = 20
        private const val DOG_INDEX = 4
        private val REQUIRED_PERMISSIONS = arrayOf(Manifest.permission.CAMERA)
        private val ANIMAL_LABELS = arrayOf("butterfly", "cat", "chicken", "cow", "dog", "elephant", "horse", "sheep", "spider", "squirrel")
    }
}

/**
 *  Rotates the bitmap image
 */
fun Bitmap.rotate(degrees: Float): Bitmap {
    val matrix = Matrix().apply { postRotate(degrees) }
    return Bitmap.createBitmap(this, 0, 0, width, height, matrix, true)
}
