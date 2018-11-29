// Distributed two-dimensional Discrete FFT transform
// Michael Martin
// ECE8893 Project 1

#include <cstdlib>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <signal.h>
#include <math.h>
#include <mpi.h>

#include "Complex.h"
#include "InputImage.h"

#define TWO_PI (M_PI * 2)

using namespace std;

void DFT(Complex* h, int w, Complex* H)
{
    // Implement a simple 1-d DFT using the double summation equation
    // given in the assignment handout.  h is the time-domain input
    // data, w is the width (N), and H is the output array.
    for (int n = 0; n < w; n++) {
        H[n] = Complex(0, 0);
        for (int k = 0; k < w; k++) {
            double exponent = TWO_PI * n * k / w;
            H[n] = H[n] + (h[k] * Complex(cos(exponent), -sin(exponent)));
        }
    }
}

void IDFT(Complex* H, int w, Complex* h)
{
    // Implement a simple 1-d IDFT using the double summation equation
    // given in the assignment handout.  h is the time-domain input
    // data, w is the width (N), and H is the output array.
    for (int n = 0; n < w; n++) {
        h[n] = Complex(0, 0);
        for (int k = 0; k < w; k++) {
            double exponent = TWO_PI * n * k / w;
            h[n] = h[n] + (H[k] * Complex(cos(exponent), sin(exponent)));
        }
        h[n] = h[n] * (1.0 / w);
    }
}

void TransformRows(int rank, int numCPUs, int width, int height, Complex *data, Complex *result, void (*transform)(Complex *h, int w, Complex *H))
{
    if (rank == 0) {

        // send rows to other cpus
        for (int cpu = 1; cpu < numCPUs; cpu++) {
            for (int row = cpu; row < height; row += numCPUs) {
                Complex *h = data + (row * width);
                int rc = MPI_Send(h, width, MPI_C_DOUBLE_COMPLEX, cpu, 0, MPI_COMM_WORLD);
                if (rc != MPI_SUCCESS) {
                    cout << "Rank " << rank << " recv failed, rc " << rc << endl;
                    MPI_Finalize();
                    exit(1);
                }
            }
        }

    } else {

        // receive rows from cpu 0
        Complex buf[width];
        for (int row = rank; row < height; row += numCPUs) {
            MPI_Status status;
            int rc = MPI_Recv(buf, width, MPI_C_DOUBLE_COMPLEX, 0, 0, MPI_COMM_WORLD, &status);
            if (rc != MPI_SUCCESS) {
                cout << "Rank " << rank << " recv failed, rc " << rc << endl;
                MPI_Finalize();
                exit(1);
            }
            copy(buf, buf + width, data + (row * width));
        }
    }

    // perform 1D transform on rows
    for (int row = rank; row < height; row += numCPUs) {
        Complex *h = data + (row * width);
        Complex *H = result + (row * width);
        transform(h, width, H);
    }

    if (rank == 0) {

        // receive row results from other cpus
        Complex buf[width];
        for (int cpu = 1; cpu < numCPUs; cpu++) {
            for (int row = cpu; row < height; row += numCPUs) {
                MPI_Status status;
                int rc = MPI_Recv(buf, width, MPI_C_DOUBLE_COMPLEX, cpu, 0, MPI_COMM_WORLD, &status);
                if (rc != MPI_SUCCESS) {
                    cout << "Rank " << rank << " recv failed, rc " << rc << endl;
                    MPI_Finalize();
                    exit(1);
                }
                copy(buf, buf + width, result + (row * width));
            }
        }

    } else {

        // send row results to cpu 0
        for (int row = rank; row < height; row += numCPUs) {
            Complex *H = result + (row * width);
            int rc = MPI_Send(H, width, MPI_C_DOUBLE_COMPLEX, 0, 0, MPI_COMM_WORLD);
            if (rc != MPI_SUCCESS) {
                cout << "Rank " << rank << " recv failed, rc " << rc << endl;
                MPI_Finalize();
                exit(1);
            }
        }
    }
}

void Transpose(Complex *array, int width, int height)
{
    for (int i = 0; i < height; i++ ) {
        for (int j = i + 1; j < width; j++ ) {
            Complex temp = array[i * width + j];
            array[i * width + j] = array[j * width + i];
            array[j * width + i] = temp;
        }
    }
}

void Transform2D(const char* inputFN) 
{ // Do the 2D transform here.
    // 1) Use the InputImage object to read in the Tower.txt file and
    //    find the width/height of the input image.
    // 2) Use MPI to find how many CPUs in total, and which one
    //    this process is
    // 3) Allocate an array of Complex object of sufficient size to
    //    hold the 2d DFT results (size is width * height)
    // 4) Obtain a pointer to the Complex 1d array of input data
    // 5) Do the individual 1D transforms on the rows assigned to your CPU
    // 6) Send the resultant transformed values to the appropriate
    //    other processors for the next phase.
    // 6a) To send and receive columns, you might need a separate
    //     Complex array of the correct size.
    // 7) Receive messages from other processes to collect your columns
    // 8) When all columns received, do the 1D transforms on the columns
    // 9) Send final answers to CPU 0 (unless you are CPU 0)
    //   9a) If you are CPU 0, collect all values from other processors
    //       and print out with SaveImageData().
    InputImage image(inputFN);  // Create the helper object for reading the image
    // Step (1) in the comments is the line above.
    // Your code here, steps 2-9

    // Step (2)
    int numCPUs, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &numCPUs);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // Step (3)
    int height, width;
    height = image.GetHeight();
    width = image.GetWidth();
    Complex result[width * height];

    // Step (4)
    Complex *data = image.GetImageData();

    // Steps (5, 6)
    TransformRows(rank, numCPUs, width, height, data, result, &DFT);
    if (rank == 0) {
        image.SaveImageData("MyAfter1D.txt", result, width, height);
        Transpose(result, width, height);
        copy(result, result + (height * width), data);
    }

    // Steps (6, 6a, 7, 8, 9, 9a)
    TransformRows(rank, numCPUs, height, width, data, result, &DFT);
    if (rank == 0) {
        Transpose(result, height, width);
        image.SaveImageData("MyAfter2D.txt", result, width, height);
    }

    // IDFT Step 1
    TransformRows(rank, numCPUs, width, height, result, data, &IDFT);
    if (rank == 0) {
        Transpose(data, width, height);
        copy(data, data + (height * width), result);
    }

    // IDFT Step 2
    TransformRows(rank, numCPUs, height, width, result, data, &IDFT);
    if (rank == 0) {
        Transpose(data, width, height);
        image.SaveImageDataReal("MyAfterInverse.txt", data, width, height);
    }
}

int main(int argc, char** argv)
{
    string fn("Tower.txt"); // default file name
    if (argc > 1) fn = string(argv[1]);  // if name specified on cmd line

    // MPI initialization here
    int rc = MPI_Init(&argc, &argv);
    if (rc != MPI_SUCCESS) {
        printf ("Error starting MPI program. Terminating.\n");
        MPI_Abort(MPI_COMM_WORLD, rc);
    }

    // Perform the transform
    Transform2D(fn.c_str());

    // Finalize MPI here
    MPI_Finalize();
    return 0;
}  
    

    
