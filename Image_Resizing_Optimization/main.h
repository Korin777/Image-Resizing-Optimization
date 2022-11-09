#pragma once

#include<iostream>
#include<string>
#include<vector>



#include <opencv\cv.hpp>
#include <GL\glew.h>
#include <glm\glm.hpp>
#include <glm\gtc\matrix_transform.hpp>
#include <glm\gtc\type_ptr.hpp>
#include <ilcplex\ilocplex.h>
#include <ilconcert\iloexpression.h>

#include "segmenatation.h"
#include "saliency.h"

using namespace std;

int target_height, target_width;


cv::Mat source_image;
cv::Mat source_image_after_smoothing;
cv::Mat source_image_after_segmentation;
cv::Mat saliency_map_of_source_image;
cv::Mat significance_map_of_source_image;


std::string source_image_file_directory = "../image/";
std::string source_image_file_name;


Graph<glm::vec2> image_graph;
std::vector<std::vector<int> > group_of_pixel;
std::vector<std::vector<double> > saliency_map;
std::vector<double> saliency_of_patch;

std::vector<cv::Vec2f> mesh_vertices;
std::vector<cv::Vec2f> target_mesh_vertices;
std::vector<double> saliency_of_mesh_vertex;

unsigned int quad_num;


float grid_size = 50;


cv::Vec3b SaliencyValueToSignifanceColor(double saliency_value) {
    // bgr
    cv::Vec3b signifance_color(0, 0, 0);

    // 理論上不會 > 1 或 < 0
    if (saliency_value > 1) {
        signifance_color[2] = 1;
        std::cout << "> 1" << std:: endl;
    }
    if (saliency_value < 0) {
        signifance_color[0] = 1;
        std::cout << "< 0" << std:: endl;
    }


    // saliency_value 越大越顯著
    if (saliency_value < (1 / 3.0)) { // B
        signifance_color[1] = (saliency_value * 3.0) * 255;
        signifance_color[0] = (1 - saliency_value * 3.0) * 255;
    }
    else if (saliency_value < (2 / 3.0)) { // G
        signifance_color[2] = ((saliency_value - (1 / 3.0)) * 3.0) * 255;
        signifance_color[1] = 1.0 * 255;
    }
    else if (saliency_value <= 1) { // R
        signifance_color[2] = 1.0 * 255;
        signifance_color[1] = (1.0 - (saliency_value - (2 / 3.0)) * 3.0) * 255;
    }

    return signifance_color;
}


// 417 * 290
void GenerateDataForImageWarping() {
    std::cout << "Start : Gaussian smoothing\n";
    const double SMOOTH_SIGMA = 0.8;
    const cv::Size K_SIZE(3, 3);

    cv::GaussianBlur(source_image, source_image_after_smoothing, K_SIZE, SMOOTH_SIGMA); // ????

    cv::imwrite(source_image_file_directory + "smooth_" + source_image_file_name, source_image_after_smoothing);
    std::cout << "Done : Gaussian smoothing\n";

    std::cout << "Start : Image segmentation\n";
    //const double SEGMENTATION_K = (source_image.size().width + source_image.size().height) / 1.75;
    const double SEGMENTATION_K = pow(source_image.size().width * source_image.size().height, 0.6); //  (width * height) ^ 0.6 = 1120.78
    const double SEGMENTATION_MIN_PATCH_SIZE = (source_image.size().width * source_image.size().height) * 0.001; // 120.93
    const double SEGMENTATION_SIMILAR_COLOR_MERGE_THRESHOLD = 20;

    source_image_after_segmentation = Segmentation(source_image, image_graph, group_of_pixel, SEGMENTATION_K, SEGMENTATION_MIN_PATCH_SIZE, SEGMENTATION_SIMILAR_COLOR_MERGE_THRESHOLD);
    cv::imwrite(source_image_file_directory + "segmentation_" + source_image_file_name + ".png", source_image_after_segmentation);
    std::cout << "Done : Image segmentation\n";

    std::cout << "Start : Image saliency calculation\n";
    const double SALIENCY_C = 3;
    const double SALIENCY_K = 64;
    saliency_map_of_source_image = CalculateContextAwareSaliencyMapWithMatlabProgram(source_image, saliency_map, source_image_file_directory + source_image_file_name, source_image_file_directory + "saliency_" + source_image_file_name + ".png");

    // Calculate the saliency value of each patch
    for (int r = 0; r < source_image.rows; ++r) {
        saliency_map[r].push_back(saliency_map[r].back());
        group_of_pixel[r].push_back(group_of_pixel[r].back());
    }
    saliency_map.push_back(saliency_map.back());
    group_of_pixel.push_back(group_of_pixel.back());

    int group_size = 0;
    for (int r = 0; r < (source_image.rows + 1); ++r) {
        for (int c = 0; c < (source_image.cols + 1); ++c) {
            group_size = std::max(group_size, group_of_pixel[r][c]);
        }
    }
    ++group_size;

    saliency_of_patch = std::vector<double>(group_size);
    std::vector<int> group_count(group_size);
    for (int r = 0; r < (source_image.rows + 1); ++r) {
        for (int c = 0; c < (source_image.cols + 1); ++c) {
            ++group_count[group_of_pixel[r][c]];
            saliency_of_patch[group_of_pixel[r][c]] += saliency_map[r][c];
        }
    }

    double min_saliency = 2e9, max_saliency = -2e9;
    for (int patch_index = 0; patch_index < group_size; ++patch_index) {
        if (group_count[patch_index]) {
            saliency_of_patch[patch_index] /= (double)group_count[patch_index];
        }
        min_saliency = std::min(min_saliency, saliency_of_patch[patch_index]);
        max_saliency = std::max(max_saliency, saliency_of_patch[patch_index]);
    }

    // Normalize saliency values
    for (int patch_index = 0; patch_index < group_size; ++patch_index) {
        saliency_of_patch[patch_index] = (saliency_of_patch[patch_index] - min_saliency) / (max_saliency - min_saliency);
    }

    significance_map_of_source_image = cv::Mat(source_image.size(), source_image.type());
    for (int r = 0; r < source_image.rows; ++r) {
        for (int c = 0; c < source_image.cols; ++c) {
            double vertex_saliency = saliency_of_patch[group_of_pixel[r][c]];
            significance_map_of_source_image.at<cv::Vec3b>(r, c) = SaliencyValueToSignifanceColor(vertex_saliency);
        }
    }
    cv::imwrite(source_image_file_directory + "significance_" + source_image_file_name + ".png", significance_map_of_source_image);

    std::cout << "Done : Image saliency calculation\n";

    //data_for_image_warping_were_generated = true;
}

void BuildGridMeshAndGraphForImage(const cv::Mat& image, Graph<glm::vec2>& target_graph) {
    target_graph = Graph<glm::vec2>();
    size_t mesh_column_count = (size_t)(image.size().width / grid_size) + 1;
    size_t mesh_row_count = (size_t)(image.size().height / grid_size) + 1;
    quad_num = (mesh_column_count-1) *(mesh_row_count-1);

    float real_mesh_width = image.size().width / (float)(mesh_column_count - 1);
    float real_mesh_height = image.size().height / (float)(mesh_row_count - 1);

    for (size_t r = 0; r < mesh_row_count; ++r) {
        for (size_t c = 0; c < mesh_column_count; ++c) {
            target_graph.vertices_.push_back(glm::vec2(c * real_mesh_width, r * real_mesh_height));
        }
    }

    //target_mesh = GLMesh();

    //target_mesh.vertices_type = GL_QUADS;

    for (size_t r = 0; r < mesh_row_count - 1; ++r) {
        for (size_t c = 0; c < mesh_column_count - 1; ++c) {
            std::vector<size_t> vertex_indices;

            size_t base_index = r * mesh_column_count + c;
            vertex_indices.push_back(base_index);
            vertex_indices.push_back(base_index + mesh_column_count);
            vertex_indices.push_back(base_index + mesh_column_count + 1);
            vertex_indices.push_back(base_index + 1);

            if (!c) {
                target_graph.edges_.push_back(Edge(std::make_pair(vertex_indices[0], vertex_indices[1])));
            }

            target_graph.edges_.push_back(Edge(std::make_pair(vertex_indices[1], vertex_indices[2])));
            target_graph.edges_.push_back(Edge(std::make_pair(vertex_indices[3], vertex_indices[2])));

            if (!r) {
                target_graph.edges_.push_back(Edge(std::make_pair(vertex_indices[0], vertex_indices[3])));
            }

            for (const size_t vertex_index : vertex_indices) {
                mesh_vertices.push_back(cv::Vec2f(target_graph.vertices_[vertex_index].x, target_graph.vertices_[vertex_index].y));
                //target_mesh.vertices_.push_back(glm::vec3(target_graph.vertices_[vertex_index].x, target_graph.vertices_[vertex_index].y, 0.0f));
                //target_mesh.uvs_.push_back(glm::vec2(target_graph.vertices_[vertex_index].x / (float)image.size().width, target_graph.vertices_[vertex_index].y / (float)image.size().height));
            }
        }
    }
}

void PatchBasedWarping(const cv::Mat& image, Graph<glm::vec2>& target_graph, const std::vector<std::vector<int> >& group_of_pixel, const std::vector<double>& saliency_of_patch, const int target_image_width, const int target_image_height, const double mesh_width, const double mesh_height) {
    if (target_image_width <= 0 || target_image_height <= 0) {
        std::cout << "Wrong target image size (" << target_image_width << " x " << target_image_height << ")\n";
        return;
    }

    // Build the edge list of each patch
    std::vector<std::vector<size_t> > edge_index_list_of_patch(image.size().width * image.size().height);
    for (size_t edge_index = 0; edge_index < target_graph.edges_.size(); ++edge_index) {
        size_t vertex_index_1 = target_graph.edges_[edge_index].edge_indices_pair_.first;
        size_t vertex_index_2 = target_graph.edges_[edge_index].edge_indices_pair_.second;

        int group_of_x = group_of_pixel[target_graph.vertices_[vertex_index_1].y][target_graph.vertices_[vertex_index_1].x];
        int group_of_y = group_of_pixel[target_graph.vertices_[vertex_index_2].y][target_graph.vertices_[vertex_index_2].x];

        if (group_of_x == group_of_y) {
            edge_index_list_of_patch[group_of_x].push_back(edge_index);
        }
        else {
            edge_index_list_of_patch[group_of_x].push_back(edge_index);
            edge_index_list_of_patch[group_of_y].push_back(edge_index);
        }
    }

    IloEnv env;

    IloNumVarArray x(env);
    IloExpr expr(env);


    // 每個 vertex 都是一個變數 v0,v1,.......,vn 
    for (size_t vertex_index = 0; vertex_index < target_graph.vertices_.size(); ++vertex_index) {
        // x,y
        x.add(IloNumVar(env, -IloInfinity, IloInfinity));
        x.add(IloNumVar(env, -IloInfinity, IloInfinity));
    }

    const double DST_WEIGHT = 5.5;
    const double DLT_WEIGHT = 0.8;
    const double ORIENTATION_WEIGHT = 12;

    const double WIDTH_RATIO = target_image_width / (double)image.size().width;
    const double HEIGHT_RATIO = target_image_height / (double)image.size().height;

    // Patch transformation constraint , ppt 15
    for (size_t patch_index = 0; patch_index < edge_index_list_of_patch.size(); ++patch_index) {
        const std::vector<size_t>& edge_index_list = edge_index_list_of_patch[patch_index];
        // patch 越大 , PATCH_SIZE_WEIGHT 越小
        const double PATCH_SIZE_WEIGHT = sqrt(1.0 / (double)edge_index_list.size());
        //const double PATCH_SIZE_WEIGHT = 1.0;

        if (!edge_index_list.size()) {
            continue;
        }



        // center edge
        //const Edge &representive_edge = target_graph.edges_[edge_index_list[edge_index_list.size() / 2]];
        const Edge& representive_edge = target_graph.edges_[edge_index_list[0]];


        double c_x = target_graph.vertices_[representive_edge.edge_indices_pair_.first].x - target_graph.vertices_[representive_edge.edge_indices_pair_.second].x;
        double c_y = target_graph.vertices_[representive_edge.edge_indices_pair_.first].y - target_graph.vertices_[representive_edge.edge_indices_pair_.second].y;


        // 反矩陣公式解 A-1 = adjA / |A|  
        // adjA => 子矩陣 → 子行列式, 正負號 → 餘因子, 轉置 → 伴隨矩陣 
        double original_matrix_a = c_x;
        double original_matrix_b = c_y;
        double original_matrix_c = c_y;
        double original_matrix_d = -c_x;

        // 行列式 |A|
        double matrix_rank = original_matrix_a * original_matrix_d - original_matrix_b * original_matrix_c;

        // 避免 |A| 太小
        if (fabs(matrix_rank) <= 1e-9) {
            matrix_rank = (matrix_rank > 0 ? 1 : -1) * 1e-9;
        }

        // 反矩陣 
        double matrix_a = original_matrix_d / matrix_rank;
        double matrix_b = -original_matrix_b / matrix_rank;
        double matrix_c = -original_matrix_c / matrix_rank;
        double matrix_d = original_matrix_a / matrix_rank;

        for (const auto& edge_index : edge_index_list) {
            const Edge& edge = target_graph.edges_[edge_index];
            double e_x = target_graph.vertices_[edge.edge_indices_pair_.first].x - target_graph.vertices_[edge.edge_indices_pair_.second].x;
            double e_y = target_graph.vertices_[edge.edge_indices_pair_.first].y - target_graph.vertices_[edge.edge_indices_pair_.second].y;

            double t_s = matrix_a * e_x + matrix_b * e_y;
            double t_r = matrix_c * e_x + matrix_d * e_y;

            // DST , similar transformation , ppt.17 , 應該是盡量讓變形與原本一致
            expr += PATCH_SIZE_WEIGHT * DST_WEIGHT * saliency_of_patch[patch_index] *
                IloPower((x[edge.edge_indices_pair_.first * 2] - x[edge.edge_indices_pair_.second * 2]) -
                    (t_s * (x[representive_edge.edge_indices_pair_.first * 2] - x[representive_edge.edge_indices_pair_.second * 2]) + t_r * (x[representive_edge.edge_indices_pair_.first * 2 + 1] - x[representive_edge.edge_indices_pair_.second * 2 + 1])),
                    2);
            expr += PATCH_SIZE_WEIGHT * DST_WEIGHT * saliency_of_patch[patch_index] *
                IloPower((x[edge.edge_indices_pair_.first * 2 + 1] - x[edge.edge_indices_pair_.second * 2 + 1]) -
                    (-t_r * (x[representive_edge.edge_indices_pair_.first * 2] - x[representive_edge.edge_indices_pair_.second * 2]) + t_s * (x[representive_edge.edge_indices_pair_.first * 2 + 1] - x[representive_edge.edge_indices_pair_.second * 2 + 1])),
                    2);

            // DLT , lenear scaling , 避免不重要的地方 over-deformation
            expr += PATCH_SIZE_WEIGHT * DLT_WEIGHT * (1 - saliency_of_patch[patch_index]) *
                IloPower((x[edge.edge_indices_pair_.first * 2] - x[edge.edge_indices_pair_.second * 2]) -
                    WIDTH_RATIO * (t_s * (x[representive_edge.edge_indices_pair_.first * 2] - x[representive_edge.edge_indices_pair_.second * 2]) + t_r * (x[representive_edge.edge_indices_pair_.first * 2 + 1] - x[representive_edge.edge_indices_pair_.second * 2 + 1])),
                    2);
            expr += PATCH_SIZE_WEIGHT * DLT_WEIGHT * (1 - saliency_of_patch[patch_index]) *
                IloPower((x[edge.edge_indices_pair_.first * 2 + 1] - x[edge.edge_indices_pair_.second * 2 + 1]) -
                    HEIGHT_RATIO * (-t_r * (x[representive_edge.edge_indices_pair_.first * 2] - x[representive_edge.edge_indices_pair_.second * 2]) + t_s * (x[representive_edge.edge_indices_pair_.first * 2 + 1] - x[representive_edge.edge_indices_pair_.second * 2 + 1])),
                    2);
        }
    }

    // Grid orientation constraint
    for (const Edge& edge : target_graph.edges_) {
        size_t vertex_index_1 = edge.edge_indices_pair_.first;
        size_t vertex_index_2 = edge.edge_indices_pair_.second;
        float delta_x = target_graph.vertices_[vertex_index_1].x - target_graph.vertices_[vertex_index_2].x;
        float delta_y = target_graph.vertices_[vertex_index_1].y - target_graph.vertices_[vertex_index_2].y;
         if (std::abs(delta_x) > std::abs(delta_y)) { // Horizontal
        expr += 0.9 * ORIENTATION_WEIGHT * IloPower(x[vertex_index_1 * 2 + 1] - x[vertex_index_2 * 2 + 1], 2);
        expr += 0.1 * ORIENTATION_WEIGHT * IloPower(x[vertex_index_1 * 2] - x[vertex_index_2 * 2], 2);
         } else {
             expr +=  0.1* ORIENTATION_WEIGHT * IloPower(x[vertex_index_1 * 2 + 1] - x[vertex_index_2 * 2 + 1], 2);
        expr +=  0.9 * ORIENTATION_WEIGHT * IloPower(x[vertex_index_1 * 2] - x[vertex_index_2 * 2], 2);
         }
    }

    IloModel model(env);

    model.add(IloMinimize(env, expr));

    IloRangeArray hard_constraint(env);

    size_t mesh_column_count = (int)(image.size().width / mesh_width) + 1;
    size_t mesh_row_count = (int)(image.size().height / mesh_height) + 1;

    // Boundary constraint
    // 最左最右 vertex 的 x keep same
    for (size_t row = 0; row < mesh_row_count; ++row) {
        size_t vertex_index = row * mesh_column_count;
        hard_constraint.add(x[vertex_index * 2] == target_graph.vertices_[0].x);

        vertex_index = row * mesh_column_count + mesh_column_count - 1;
        hard_constraint.add(x[vertex_index * 2] == target_graph.vertices_[0].x + target_image_width - 1);
    }
    // 最上最下 vertex 的 y keep same
    for (size_t column = 0; column < mesh_column_count; ++column) {
        size_t vertex_index = column;
        hard_constraint.add(x[vertex_index * 2 + 1] == target_graph.vertices_[0].y);

        vertex_index = (mesh_row_count - 1) * mesh_column_count + column;
        hard_constraint.add(x[vertex_index * 2 + 1] == target_graph.vertices_[0].y + target_image_height - 1);
    }

    // Avoid 
    // 應該是要讓 vetex 的順序不變 , 也就是 vertex 相對位置不變
    for (size_t row = 0; row < mesh_row_count; ++row) {
        for (size_t column = 1; column < mesh_column_count; ++column) {
            size_t vertex_index_right = row * mesh_column_count + column;
            size_t vertex_index_left = row * mesh_column_count + column - 1;
            hard_constraint.add((x[vertex_index_right * 2] - x[vertex_index_left * 2]) >= 1e-4);
        }
    }

    for (size_t row = 1; row < mesh_row_count; ++row) {
        for (size_t column = 0; column < mesh_column_count; ++column) {
            size_t vertex_index_down = row * mesh_column_count + column;
            size_t vertex_index_up = (row - 1) * mesh_column_count + column;
            hard_constraint.add((x[vertex_index_down * 2 + 1] - x[vertex_index_up * 2 + 1]) >= 1e-4);
        }
    }

    model.add(hard_constraint);

    IloCplex cplex(model);

    cplex.setOut(env.getNullStream());
    if (!cplex.solve()) {
        std::cout << "Failed to optimize the model.\n";
    }

    IloNumArray result(env);

    cplex.getValues(result, x);

    for (size_t vertex_index = 0; vertex_index < target_graph.vertices_.size(); ++vertex_index) {
        target_graph.vertices_[vertex_index].x = result[vertex_index * 2];
        target_graph.vertices_[vertex_index].y = result[vertex_index * 2 + 1];
    }

    model.end();
    cplex.end();
    env.end();
}

void ContentAwareImageRetargeting(const int target_image_width, const int target_image_height) {
    if (!source_image.size().width || !source_image.size().height) {
        return;
    }
    // Segmentation, saliency, significant
    GenerateDataForImageWarping();




    // ok

    std::cout << "Start : Build mesh and graph\n";
    BuildGridMeshAndGraphForImage(source_image, image_graph);
    std::cout << "Done : Build mesh and graph\n";


    //if (program_mode == PATCH_BASED_WARPING) {

    std::cout << "Start : Patch based warping\n";
    PatchBasedWarping(source_image, image_graph, group_of_pixel, saliency_of_patch, target_image_width, target_image_height, grid_size, grid_size);
    std::cout << "Done : Patch based warping\n";

    std::cout << "New image size : " << target_image_width << " " << target_image_height << "\n";
    //} else if (program_mode == FOCUS_WARPING) {
    //  std::cout << "Start : Focus warping\n";
    //  FocusWarping(image, image_graph, group_of_pixel, saliency_of_patch, target_image_width, target_image_height, mesh_width, mesh_height, focus_mesh_scale, focus_x, focus_y);
    //  std::cout << "Done : Focus warping\n";
    //  std::cout << "New image size : " << target_image_width << " " << target_image_height << "\n";
    //}

    //saliency_of_mesh_vertex.clear();
    //saliency_of_mesh_vertex = std::vector<double>(image_graph.vertices_.size());

    //for (size_t vertex_index = 0; vertex_index < image_graph.vertices_.size(); ++vertex_index) {
    //  float original_x = gl_panel_image_mesh.vertices_[vertex_index].x;
    //  float original_y = gl_panel_image_mesh.vertices_[vertex_index].y;
    //  saliency_of_mesh_vertex[vertex_index] = saliency_of_patch[group_of_pixel[original_y][original_x]];
    //}

    int mesh_column_count = (int)(source_image.size().width / grid_size) + 1;
    int mesh_row_count = (int)(source_image.size().height / grid_size) + 1;
    int target_mesh_column_count = (int)(target_image_width / grid_size);
    int target_mesh_row_count = (int)(target_image_height / grid_size);

    float real_mesh_width = source_image.size().width / (float)(mesh_column_count - 1);
    float real_mesh_height = source_image.size().height / (float)(mesh_row_count - 1);

    //gl_panel_image_mesh.vertices_.clear();

    for (int r = 0; r < mesh_row_count - 1; ++r) {
      for (int c = 0; c < mesh_column_count - 1; ++c) {
          std::vector<size_t> vertex_indices;

          size_t base_index = r * (mesh_column_count)+c;
          vertex_indices.push_back(base_index);
          vertex_indices.push_back(base_index + mesh_column_count);
          vertex_indices.push_back(base_index + mesh_column_count + 1);
          vertex_indices.push_back(base_index + 1);

          for (const auto& vertex_index : vertex_indices) {
              if (image_graph.vertices_[vertex_index].x == 633)
                  cout << "fuck" << endl;
              target_mesh_vertices.push_back(cv::Vec2f(image_graph.vertices_[vertex_index].x, image_graph.vertices_[vertex_index].y));
              //gl_panel_image_mesh.vertices_.push_back(glm::vec3(image_graph.vertices_[vertex_index].x, image_graph.vertices_[vertex_index].y, 0));
          }
      }
    }

    //saliency_of_mesh_vertex = std::vector<double>(image_graph.vertices_.size());

    //for (size_t vertex_index = 0; vertex_index < image_graph.vertices_.size(); ++vertex_index) {
    //    float original_x = gl_panel_image_mesh.vertices_[vertex_index].x;
    //    float original_y = gl_panel_image_mesh.vertices_[vertex_index].y;
    //    saliency_of_mesh_vertex[vertex_index] = saliency_of_patch[group_of_pixel[original_y][original_x]];
    //}

    cv::Mat result_image(cv::Size2d(target_image_width, target_image_height), CV_8UC4, cv::Scalar(0, 0, 0, 0));
    cv::Mat result_image3(cv::Size2d(target_image_width, target_image_height), CV_8UC3, cv::Scalar(0, 0, 0));
    cv::Mat source(source_image.size(), CV_8UC4, cv::Scalar(0, 0, 0, 0));
    cv::Mat mask(cv::Size2d(target_image_width, target_image_height), CV_8UC1, cv::Scalar::all(0));

    cv::Mat test(cv::Size2d(target_image_width, target_image_height), CV_8UC4, cv::Scalar(0, 0, 0, 0));

    unsigned int n = 0;
    for (int i = 0; i < quad_num; i++) {
        cv::Point2f src[4], dst[4];

        cv::Mat result(target_image_width, target_image_height, CV_8UC4, cv::Scalar(0, 0, 0, 0));
        cv::Mat mask1(source_image.size(), CV_8UC1, cv::Scalar::all(0));
        cv::Mat black1(source_image.size(), source_image.type(), cv::Scalar(0, 0, 0));

        vector<vector<cv::Point>> contour;
        contour.push_back(vector<cv::Point>());

        for (int j = 0; j < 4; j++) {
            src[j].x = mesh_vertices[n][0];
            src[j].y = mesh_vertices[n][1];
            dst[j].x = target_mesh_vertices[n][0];
            dst[j].y = target_mesh_vertices[n][1];
            // 跳過重複的邊 
            /*
                0-3
                | |
                1-2
            */
            if (i % (mesh_column_count - 1) != 0 && j < 2) {
                src[j].x += 1;
                //dst[j].x += 1;
            }
            if (i >= (mesh_column_count - 1) != 0 && (j == 0 || j == 3)) {
                src[j].y += 1;
                //dst[j].y += 1;
            }

            n++;

            contour[0].push_back(cv::Point(src[j].x, src[j].y));
        }
        cv::drawContours(mask1, contour, 0, cv::Scalar(255, 255, 255), cv::FILLED);
        source_image.copyTo(black1, mask1);


        cv::cvtColor(black1, black1, cv::COLOR_BGR2BGRA);

        //if (i < 12) {
        //    //imshow("tmp" + char(i), black1);
        //    cv::add(test, black1, test);
        //}

        //if (i == 13) {
        //    for (int k = 0; k < source_image.size().height; ++k) {
        //        test.at<cv::Vec4b>(k, 52) = (255, 0, 0, 255);
        //    }
        //    for (int k = 0; k < source_image.size().height; ++k) {
        //        cout << test.at<cv::Vec4b>(k, 52);
        //    }
        //}

        cv::Mat perspectiveTransform = cv::getPerspectiveTransform(src, dst);
        cv::warpPerspective(black1, result, perspectiveTransform, cv::Size2d(target_image_width, target_image_height), 1, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0, 0));
        
        /*if (i == 0) {
            cout << contour[0];
            for(int j = 0; j < 30; ++j)
                cout << result.at<cv::Vec4b>(0, j) << ' ';
            cout << endl;
        }
        if (i == 1) {
            cout << contour[0];
            for (int j = 0; j < 30; ++j)
                cout << result.at<cv::Vec4b>(0, j) << ' ';
        }*/

        // 下邊界
        cv::Vec3b significance_color = SaliencyValueToSignifanceColor(saliency_of_patch[group_of_pixel[src[1].y][src[1].x]]);
        for (int k = dst[1].x; k <= dst[2].x; ++k) {
            for (int j = min(dst[1].y,dst[2].y)/*dst[0].y*/;j < target_image_height-1; ++j) {
                auto tmp = result.at<cv::Vec4b>(j+1, k);
                if (tmp[0] == 0 && tmp[1] == 0 && tmp[2] == 0) {
                    result.at<cv::Vec4b>(j - 1, k)[0] = significance_color[0];
                    result.at<cv::Vec4b>(j - 1, k)[1] = significance_color[1];
                    result.at<cv::Vec4b>(j - 1, k)[2] = significance_color[2];
                    result.at<cv::Vec4b>(j - 1, k)[3] = 255;
                    result.at<cv::Vec4b>(j, k)[0] = significance_color[0];
                    result.at<cv::Vec4b>(j, k)[1] = significance_color[1];
                    result.at<cv::Vec4b>(j, k)[2] = significance_color[2];
                    result.at<cv::Vec4b>(j, k)[3] = 255;
 /*                   result.at<cv::Vec4b>(j+1, k)[0] = significance_color[0];
                    result.at<cv::Vec4b>(j+1, k)[1] = significance_color[1];
                    result.at<cv::Vec4b>(j+1, k)[2] = significance_color[2];
                    result.at<cv::Vec4b>(j+1, k)[3] = 255;*/
                    mask.at<uchar>(j - 1, k) = 255;
                    mask.at<uchar>(j, k) = 255;
                    //mask.at<uchar>(j+1, k) = 255;
                    //cout << j << ' ' << k << ' ' << result.at<cv::Vec4b>(j, k) << endl;
                    break;
                }
            }
        }

         //右邊界
        significance_color = SaliencyValueToSignifanceColor(saliency_of_patch[group_of_pixel[src[3].y][src[3].x]]);
        for (int j = dst[3].y; j <= dst[2].y; ++j) {
            for (int k = min(dst[2].x,dst[3].x)/*dst[0].x*/; k < target_image_width - 1; ++k) {
                auto tmp = result.at<cv::Vec4b>(j, k + 1);
                if (tmp[0] == 0 && tmp[1] == 0 && tmp[2] == 0) {
                    result.at<cv::Vec4b>(j, k)[0] = significance_color[0];
                    result.at<cv::Vec4b>(j, k)[1] = significance_color[1];
                    result.at<cv::Vec4b>(j, k)[2] = significance_color[2];
                    result.at<cv::Vec4b>(j, k)[3] = 255;
                    result.at<cv::Vec4b>(j, k-1)[0] = significance_color[0];
                    result.at<cv::Vec4b>(j, k-1)[1] = significance_color[1];
                    result.at<cv::Vec4b>(j, k-1)[2] = significance_color[2];
                    result.at<cv::Vec4b>(j, k-1)[3] = 255;
                    mask.at<uchar>(j, k) = 255;
                    mask.at<uchar>(j, k-1) = 255;
                    break;
                }
            }
        }

        for (int j = 0; j < target_image_height; ++j) {
           for (int k = 0; k < target_image_width; ++k) {
               cv::Vec4b tmp = result_image.at<cv::Vec4b>(j, k);
               cv::Vec4b tmp2 = result.at<cv::Vec4b>(j, k);
               if (!(tmp2[0] == 0 && tmp2[1] == 0 && tmp2[2] == 0)) {
                   if (tmp[0] == 0 && tmp[1] == 0 && tmp[2] == 0) {
                       result_image.at<cv::Vec4b>(j, k) = result.at<cv::Vec4b>(j, k);
                   }
                   //else {
                   //    int rgb = (int)tmp[0] + (int)tmp[1] + (int)tmp[2], rgb2 = (int)tmp2[0] + (int)tmp2[1] + (int)tmp2[2];

                   //    if (rgb < rgb2) {
                   //        //cout << j << ' ' << k << endl;
                   //        result_image.at<cv::Vec4b>(j, k) = result.at<cv::Vec4b>(j, k);
                   //    }
                   //}
               }
           }
       }
        
        
        //for (int j = 0; j < target_image_height; ++j) {
        //    for (int k = 0; k < target_image_width; ++k) {
        //        cv::Vec4b tmp = result_image.at<cv::Vec4b>(j, k);
        //        cv::Vec4b tmp2 = result.at<cv::Vec4b>(j, k);
        //        if (tmp2[0] != 0 && tmp2[1] != 0 && tmp2[2] != 0) {
        //            if (tmp[0] == 0 && tmp[1] == 0 && tmp[2] == 0) {
        //                result_image.at<cv::Vec4b>(j, k) = result.at<cv::Vec4b>(j, k);
        //            }
        //            else {
        //                int rgb = (int)tmp[0] + (int)tmp[1] + (int)tmp[2], rgb2 = (int)tmp2[0] + (int)tmp2[1] + (int)tmp2[2];

        //                if (rgb < rgb2) {
        //                    //cout << j << ' ' << k << endl;
        //                    result_image.at<cv::Vec4b>(j, k) = result.at<cv::Vec4b>(j, k);
        //                }
        //            }
        //        }
        //    }
        //}
        //if (i < 12) {
        //    //imshow("tmp" + char(i), black1);
        //    cv::add(test, result, test);
        //}

 /*       if (i == 13) {
            for (int k = 0; k < source_image.size().height; ++k) {
                test.at<cv::Vec4b>(k, 52) = (255, 0, 0, 255);
            }
            for (int k = 0; k < source_image.size().height; ++k) {
                cout << test.at<cv::Vec4b>(k, 52);
            }
        }*/
        
        //cv::add(result_image, result, result_image);
        cv::add(source, black1, source);
    }
    
    for (int i = 0; i < target_image_height; ++i) {
        for (int j = 0; j < target_image_width; ++j) {
            for (int k = 0; k < 3; ++k)
                result_image3.at<cv::Vec3b>(i, j)[k] = result_image.at<cv::Vec4b>(i, j)[k];
        }
    }

    cv::Mat inpaint_result(cv::Size2d(target_image_width, target_image_height), CV_8UC3, cv::Scalar(0, 0, 0));
    cv::inpaint(result_image3, mask, inpaint_result, 3, cv::INPAINT_TELEA);
    imshow("result3", inpaint_result);
    cv::imwrite(source_image_file_directory + "inpaint.png", inpaint_result);
    imshow("source", source_image);
    cv::resize(source_image, source_image, cv::Size2d(target_image_width, target_image_height));
    imshow("resize", source_image);
    cv::imwrite(source_image_file_directory + "linear.png", source_image);

    cv::namedWindow("Result");
    cv::Mat show_result = result_image.clone();
    imshow("Result", show_result);
    cv::imwrite(source_image_file_directory + "result.png", result_image);


}