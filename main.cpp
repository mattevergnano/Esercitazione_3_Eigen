#include <iostream>
#include "Eigen/Eigen"
#include "Eigen/Dense"
#include<iomanip>
using namespace std;
using namespace Eigen;

Vector2d elleu(Matrix2d A, Vector2d b){
    Vector2d x = Vector2d::Zero();
    Vector2d y = Vector2d::Zero();
    PartialPivLU<Matrix2d> lu(A);
    Matrix2d P = lu.permutationP();
    Matrix2d L = Matrix2d::Identity(2,2);
    L.triangularView<StrictlyLower>() = lu.matrixLU();
    Matrix2d U = lu.matrixLU().triangularView<Upper>();
    y = (L.inverse()* P) * b;
    x = U.inverse() * y;
    return x;
}

Vector2d qerre(Matrix2d A, Vector2d b){
    Vector2d x = Vector2d::Zero();
    Vector2d y = Vector2d::Zero();
    HouseholderQR<Matrix2d> qr = A.householderQr();
    MatrixXd R = qr.matrixQR().triangularView<Eigen::Upper>();
    MatrixXd Q = qr.householderQ();
    y = Q.inverse() * b;
    x = R.inverse() * y;
    return x;
}


int main()
{
    Vector2d sol = Vector2d::Zero();
    sol << -1.0e+00, -1.0e+00;
    Matrix2d A1 = Matrix2d::Zero();
    Vector2d V1 = Vector2d::Zero();
    A1 << 5.547001962252291e-01,-3.770900990025203e-02,8.320502943378437e-01,-9.992887623566787e-01;
    V1 << -5.169911863249772e-01, 1.672384680188350e-01;
    Matrix2d A2 = Matrix2d::Zero();
    Vector2d V2 = Vector2d::Zero();
    A2 << 5.547001962252291e-01, -5.540607316466765e-01, 8.320502943378437e-01,-8.324762492991313e-01;
    V2 << -6.394645785530173e-04, 4.259549612877223e-04;
    Matrix2d A3 = Matrix2d::Zero();
    Vector2d V3 = Vector2d::Zero();
    A3 << 5.547001962252291e-01, -5.547001955851905e-01, 8.320502943378437e-01,-8.320502947645361e-01;
    V3 << -6.400391328043042e-10, 4.266924591433963e-10;
    Vector2d x1LU = Vector2d::Zero();
    x1LU = elleu(A1,V1);
    double err1LU = (x1LU-sol).norm() / sol.norm();
    cout << "Errore relativo 1 LU: " << err1LU << endl;
    Vector2d x2LU = Vector2d::Zero();
    x2LU = elleu(A2,V2);
    double err2LU = (x2LU-sol).norm() / sol.norm();
    cout << "Errore relativo 2 LU: " << err2LU << endl;
    Vector2d x3LU = Vector2d::Zero();
    x3LU = elleu(A3,V3);
    double err3LU = (x3LU-sol).norm() / sol.norm();
    cout << "Errore relativo 3 LU: " << err3LU << endl;

    Vector2d x1QR = Vector2d::Zero();
    x1QR = qerre(A1,V1);
    double err1QR = (x1QR-sol).norm() / sol.norm();
    cout << "Errore relativo 1 QR: " << err1QR << endl;
    Vector2d x2QR = Vector2d::Zero();
    x2QR = qerre(A2,V2);
    double err2QR = (x2QR-sol).norm() / sol.norm();
    cout << "Errore relativo 2 QR: " << err2QR << endl;
    Vector2d x3QR = Vector2d::Zero();
    x3QR = qerre(A3,V3);
    double err3QR = (x3QR-sol).norm() / sol.norm();
    cout << "Errore relativo 3 QR: " << err3QR << endl;

    return 0;
}
