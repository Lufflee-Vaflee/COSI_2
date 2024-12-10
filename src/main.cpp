#include <opencv2/opencv.hpp>
#include <iostream>
#include <random>

cv::Mat getHistogram(cv::Mat const& image);
cv::Mat gammaCorrection(const cv::Mat& img, double gamma, double c);
cv::Mat harmonicAverage(cv::Mat const& img);
cv::Mat addImpulseNoise(cv::Mat const& image, double noiseRatio);
cv::Mat applyHarmonicMedianFilter(const cv::Mat& image, int neighborhoodSize = 3);

int main(int argc, char* argv[]) {
    std::string imagePath = argv[1];  // Replace with the path to your image
    double c = 1;
    double y = 1;

    if(argc >= 3) {
        c = std::stof(argv[2]);
    }

    if(argc >= 4) {
        y = std::stof(argv[3]);
    }

    // Load the image from file
    cv::Mat image = cv::imread(imagePath);
    if (image.empty()) {
        std::cerr << "Error: Could not load image!" << std::endl;
        return -1;
    }
    std::cout << image.channels() << std::endl;

    cv::Mat gammaCorrected = gammaCorrection(image, y, c);
    cv::Mat noised = addImpulseNoise(image, 0.05);
    cv::Mat average = harmonicAverage(image);
    cv::Mat median = applyHarmonicMedianFilter(noised);


    cv::imshow("Original", image);
    cv::imshow("Original Histogramm", getHistogram(image));
    cv::imshow("Gamma corrected", gammaCorrected);
    cv::imshow("Gamma corrected Histogramm", getHistogram(gammaCorrected));


    cv::imshow("Original", image);
    cv::imshow("Harmonic average", average);
    cv::imshow("Noised", noised);
    cv::imshow("Median", median);

    // Close all OpenCV windows
    while(true) {
        cv::waitKey();
    }

    return 0;
}

cv::Mat gammaCorrection(const cv::Mat& img, double y, double c) {
    CV_Assert(img.data); // Ensure image is not empty

    // Convert image to float type for processing
    cv::Mat imgFloat;
    img.convertTo(imgFloat, CV_32F, 1.0 / 255.0); // Normalize to [0, 1]

    cv::Mat imgGammaCorrected = c * cv::Mat::zeros(imgFloat.size(), imgFloat.type());
    cv::pow(imgFloat, y, imgGammaCorrected); // I^gamma
    imgGammaCorrected *= c; // Multiply by constant c

    // Convert back to 8-bit format
    imgGammaCorrected.convertTo(imgGammaCorrected, CV_8U, 255.0);

    return imgGammaCorrected;
}

template<typename el = cv::Vec3b>
std::vector<el> getPixelNeighbors(const cv::Mat& image, int x, int y, int neighborhoodSize = 3) {
    std::vector<el> neighbors;

    if (image.empty() || x < 0 || y < 0 || x >= image.cols || y >= image.rows) {
        std::cerr << "Invalid parameters or image." << std::endl;
        return neighbors;
    }

    int halfSize = neighborhoodSize / 2;

    for (int dy = -halfSize; dy <= halfSize; ++dy) {
        for (int dx = -halfSize; dx <= halfSize; ++dx) {
            int nx = x + dx;
            int ny = y + dy;

            if (nx >= 0 && nx < image.cols && ny >= 0 && ny < image.rows) {
                neighbors.push_back(image.at<el>(ny, nx));
            }
        }
    }

    return neighbors;
}

auto calculateHarmonic(std::vector<cv::Vec3b> const& neighbours) {
    double accumulate1 = std::numeric_limits<double>::min();
    double accumulate2 = std::numeric_limits<double>::min();
    double accumulate3 = std::numeric_limits<double>::min();

    for(std::size_t i = 0; i < neighbours.size(); ++i) {
        if(neighbours[i][0] != 0)
            accumulate1 += (1 / static_cast<double>(neighbours[i][0]));
        if(neighbours[i][1] != 0)
            accumulate2 += (1 / static_cast<double>(neighbours[i][1]));
        if(neighbours[i][2] != 0)
            accumulate3 += (1 / static_cast<double>(neighbours[i][2]));
    }

    double count = static_cast<double>(neighbours.size());
    cv::Vec3b result{
        static_cast<uchar>(count / accumulate1),
        static_cast<uchar>(count / accumulate2),
        static_cast<uchar>(count / accumulate3),
    };

    return result;
}

auto calculateHarmonic(std::vector<uchar> const& neighbours) {
    double accumulate1 = std::numeric_limits<double>::min();
    for(std::size_t i = 0; i < neighbours.size(); ++i) {
        accumulate1 += (1 / (neighbours[i] + std::numeric_limits<double>::min()));
    }

    return neighbours.size() / accumulate1;
}

cv::Mat harmonicAverage(cv::Mat const& img) {
    CV_Assert(img.data); // Ensure image is not empty
    cv::Mat result;
    img.copyTo(result);
    if(img.channels() == 3) {
        for (int i = 1; i < img.rows - 1; ++i) {
            for (int j = 1; j < img.cols - 1; ++j) {
                auto neighbours = getPixelNeighbors(img, j, i);
                auto after = calculateHarmonic(neighbours);
                result.at<cv::Vec3b>(i, j) = after;
            }
        }
    } else {
        for (int i = 1; i < img.rows - 1; ++i) {
            for (int j = 1; j < img.cols - 1; ++j) {
                auto neighbours = getPixelNeighbors<uchar>(img, j, i);
                result.at<uchar>(i, j) = calculateHarmonic(neighbours);
            }
        }
    }

    return result;
}

// Function to compute harmonic median for a single channel
uchar harmonicMedianSingleChannel(const std::vector<uchar>& values) {
    std::vector<unsigned> reciprocals;

    for (uchar value : values) {
        reciprocals.push_back(static_cast<unsigned>(value));
    }

    // Sort reciprocals
    std::sort(reciprocals.begin(), reciprocals.end());

    return reciprocals[4];
}

// Function to compute harmonic median for color pixels
cv::Vec3b harmonicMedian(const std::vector<cv::Vec3b>& neighbors) {
    std::vector<uchar> channel1, channel2, channel3;

    // Split neighbors into individual channels
    for (const auto& neighbor : neighbors) {
        channel1.push_back(neighbor[0]);
        channel2.push_back(neighbor[1]);
        channel3.push_back(neighbor[2]);
    }

    return cv::Vec3b(
        harmonicMedianSingleChannel(channel1),
        harmonicMedianSingleChannel(channel2),
        harmonicMedianSingleChannel(channel3)
    );
}

// Apply harmonic median filter to an image
cv::Mat applyHarmonicMedianFilter(const cv::Mat& image, int neighborhoodSize) {
    cv::Mat result = image.clone();

    for (int y = 1; y < image.rows - 1; ++y) {
        for (int x = 1; x < image.cols - 1; ++x) {
            if(image.channels() == 3) {
                auto neighbors = getPixelNeighbors(image, x, y, neighborhoodSize);
                result.at<cv::Vec3b>(y, x) = harmonicMedian(neighbors);
            } else {
                auto neighbors = getPixelNeighbors<uchar>(image, x, y, neighborhoodSize);
                result.at<uchar>(y, x) = harmonicMedianSingleChannel(neighbors);
            }
        }
    }

    return result;
}

cv::Mat addImpulseNoise(cv::Mat const& image, double noiseRatio) {
    cv::Mat result;
    image.copyTo(result);
    if (result.empty() || noiseRatio <= 0 || noiseRatio > 1) {
        std::cerr << "Некорректные входные параметры." << std::endl;
        return result;
    }

    // Проверяем, что изображение в правильном формате
    if (result.channels() != 1 && result.channels() != 3) {
        std::cerr << "Только одноканальные и трехканальные изображения поддерживаются." << std::endl;
        return result;
    }

    // Генератор случайных чисел
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dist(0, 1);

    // Количество пикселей, которые нужно заменить на шум
    int totalPixels = result.rows * result.cols;
    int noisyPixels = static_cast<int>(totalPixels * noiseRatio);

    for (int i = 0; i < noisyPixels; ++i) {
        // Генерация случайных координат
        int x = std::uniform_int_distribution<>(0, result.cols - 1)(gen);
        int y = std::uniform_int_distribution<>(0, result.rows - 1)(gen);

        if (result.channels() == 1) {
            // Для одноканального изображения
            result.at<uchar>(y, x) = (dist(gen) > 0.5) ? 255 : 0;
        } else if (result.channels() == 3) {
            // Для трехканального изображения
            cv::Vec3b& pixel = result.at<cv::Vec3b>(y, x);
            if (dist(gen) > 0.5) {
                pixel = cv::Vec3b(255, 255, 255); // белый
            } else {
                pixel = cv::Vec3b(0, 0, 0);       // черный
            }
        }
    }

    return result;
}

cv::Mat getHistogram(cv::Mat const& image) {
    cv::Mat grayImage;
    if(image.channels() == 3) {
        cv::cvtColor(image, grayImage, cv::COLOR_BGR2GRAY);
    } else {
        grayImage = image;
    }

    int histSize = 256;  // Количество бинов (градаций яркости)
    float range[] = {0, 256};  // Диапазон значений пикселей
    const float* histRange = {range};

    cv::Mat hist;
    cv::calcHist(&grayImage, 1, 0, cv::Mat(), hist, 1, &histSize, &histRange);

    // Нормализуем гистограмму (для лучшего отображения)
    cv::Mat histNorm;
    cv::normalize(hist, histNorm, 0, 400, cv::NORM_MINMAX);  // Нормализация с учетом высоты изображения

    // Создаем изображение для отображения гистограммы
    int histWidth = 512;
    int histHeight = 400;
    cv::Mat histImage(histHeight, histWidth, CV_8UC1, cv::Scalar(255));  // Белый фон

    // Рисуем гистограмму
    for (int i = 1; i < histSize; i++) {
        int x1 = (i - 1) * histWidth / histSize;
        int y1 = histHeight - cvRound(histNorm.at<float>(i - 1));  // Корректируем высоту
        int x2 = i * histWidth / histSize;
        int y2 = histHeight - cvRound(histNorm.at<float>(i));  // Корректируем высоту

        // Убедимся, что y1 и y2 не выходят за пределы
        y1 = std::max(y1, 0);
        y2 = std::max(y2, 0);

        // Рисуем линию для текущего диапазона яркости
        cv::line(histImage,
                 cv::Point(x1, y1),
                 cv::Point(x2, y2),
                 cv::Scalar(0, 0, 255), 2);  // Красный цвет для линии
    }

    return histImage;
}
