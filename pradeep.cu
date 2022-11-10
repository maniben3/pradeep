#include <iostream>

using namespace std;

extern "C" {
  __host__ int other(int n)
  {
          for(int i=0; i<n; i++)
                  cout << "Hello, world!" << endl;

          return 0;
  }
}
