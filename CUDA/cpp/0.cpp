#include<bits/stdc++.h>
using namespace std;
typedef chrono::steady_clock clk;
// #define us std::chrono::duration_cast<std::chrono::milliseconds>(chrono::steady_clock::now().time_since_epoch()).count()
#define N 20
int a[int(1<<N)+10][N + 1];
int b[N + 1][int(1<<N)+10];

int main(){
    ios::sync_with_stdio(0); cin.tie(0);
    clk::time_point st = clk::now(), mid;
    int cou = 0;
    for(int i = 0; i < (1<<N); i++){
        for(int j = 0; j < N; j++){
            a[i][j] = (i>>j)&1;
        }
    }
    mid = clk::now();
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < (1 << N); j++) {
            b[i][j] = (j >> i) % 2;
        }
    }
    clk::time_point en = clk::now();

    cout << chrono::duration_cast<chrono::milliseconds>(mid - st).count() << "ms\n";
    cout << chrono::duration_cast<chrono::microseconds>(mid - st).count() << "us\n";
    cout << chrono::duration_cast<chrono::milliseconds>(en - mid).count() << "ms\n";
    cout << chrono::duration_cast<chrono::microseconds>(en - mid).count() << "us\n";
    // cout << cou << "\n";
    return 0;
}