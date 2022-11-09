#pragma once


#include <vector>
#include <algorithm>
#include <set>
#include <map>

#include <ilcplex\ilocplex.h>
#include <ilconcert\iloexpression.h>
#include <opencv\cv.hpp>
#include <glm\glm.hpp>
#include <glm\gtc\matrix_transform.hpp>
#include <glm\gtc\type_ptr.hpp>

#include "graph.h"

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
    const double ORIENTATION_WEIGHT = 12.0;

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
        // if (std::abs(delta_x) > std::abs(delta_y)) { // Horizontal
        expr += ORIENTATION_WEIGHT * IloPower(x[vertex_index_1 * 2 + 1] - x[vertex_index_2 * 2 + 1], 2);
        // } else {
        expr += ORIENTATION_WEIGHT * IloPower(x[vertex_index_1 * 2] - x[vertex_index_2 * 2], 2);
        // }
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
        hard_constraint.add(x[vertex_index * 2] == target_graph.vertices_[0].x + target_image_width);
    }
    // 最上最下 vertex 的 y keep same
    for (size_t column = 0; column < mesh_column_count; ++column) {
        size_t vertex_index = column;
        hard_constraint.add(x[vertex_index * 2 + 1] == target_graph.vertices_[0].y);

        vertex_index = (mesh_row_count - 1) * mesh_column_count + column;
        hard_constraint.add(x[vertex_index * 2 + 1] == target_graph.vertices_[0].y + target_image_height);
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