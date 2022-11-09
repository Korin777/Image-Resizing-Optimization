#include "main.h"
using namespace std;

int main() {
   
    cout << "input image:";
    cin >> source_image_file_name;
    cout << "height:";
    cin >> target_height;
    cout << "width:";
    cin >> target_width;
    cout << "grid:";
    cin >> grid_size; // degault size = 50

	source_image = cv::imread(source_image_file_directory + source_image_file_name);
    ContentAwareImageRetargeting(target_width, target_height);


	cv::waitKey(0);

}