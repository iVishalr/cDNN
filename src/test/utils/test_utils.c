#include "../../utils/utils.h"
#include "./test_utils.h"

void TEST_DOT(){
  FILE * fp;
  char * line = NULL;
  size_t len = 0;
  ssize_t read;
  dARRAY * matrix = NULL;
  dARRAY * result = NULL;
  int successes = 0;
  int isValid = 0;
  fp = fopen("./src/test/TEST_DOT.txt", "r");
  if (fp == NULL)
      exit(EXIT_FAILURE);
  int count=0;
  while ((read = getline(&line, &len, fp)) != -1){
    matrix = pre_process(line);
    printf("Matrix constructed : Shape = ");
    shape(matrix);
    result = perform_dot(matrix);
    printf("Dot Product constructed : Shape = ");
    shape(result);
    isValid = validate(result);
    if(!isValid){
      printf("ERROR : Obtained = ");
      for(int i=0;i<result->shape[0]*result->shape[1];i++){
        printf("%d",(int)result->matrix[i]);
      }
      printf(". Expected Result is different.\n");
    }
    successes += isValid;
    if(count==6) break;
    count++;
    free2d(matrix);
    free2d(result);
    matrix = NULL;
    result = NULL;
  }
  printf("Report : \nTotal Number of Correct Dot Operations : %d\n",successes);
  printf("Score : %f %%\n",successes*100/10.0);
  fclose(fp);
  if (line){
      free(line);
      line = NULL;
  }
}

void sleep(int milliseconds) {
  //Function to create a time delay. Mimicks thread.sleep() of Java
  unsigned int duration = time(0) + (milliseconds/1000);
  while(time(0)<duration);
}

dARRAY * pre_process(char * test){  
  char * temp = (char*)malloc(sizeof(char)*5000);
  char * shape = (char*)malloc(sizeof(char)*100);
  int i=0;
  int j=0;
  int k=0;
  int flag = 0;
  while(test[k]!='\n'){
    if(test[k]==' '){
      flag=1;
    }
    else if(flag){
      if(test[k]!='['&& test[k]!=']'){
        shape[j] = test[k];
        j++;
      }
    }
    else if(!flag){
      temp[i] = test[k];
      i++;
    }
    k++;
  }
  temp[i]='\0';
  shape[j] = '\0';
  int dims[2];
  //extract shape from the shape string
  get_shape_arr(shape,dims);
  free(shape);
  shape = NULL;
  //now we need to extract digits from temp and create dArray obj
  dARRAY * matrix = NULL;
  matrix = (dARRAY*)malloc(sizeof(dARRAY));
  matrix->matrix = (double*)malloc(sizeof(double)*dims[0]*dims[1]);
  matrix->shape[0] = dims[0];
  matrix->shape[1] = dims[1];
  get_martix_arr(matrix,temp);
  // printf("Finished building the matrix!\n");
  free(temp);
  temp = NULL;
  return matrix;
}

void get_shape_arr(char * shape,int * dims){
  int i=0;
  char * token = strtok(shape, ",");
  const char * temp[2];
  //loop through the string to extract all other tokens
  while(token != NULL && i<2){
    temp[i] = token;
    i++;
    token = strtok(NULL, ",");
  }
  dims[0] = atoi(temp[0]);
  dims[1] = atoi(temp[1]);
  return;
}

void get_martix_arr(dARRAY * matrix, char * test){
  int i=0;
  while(test[i]!='\0'){
    matrix->matrix[i] = (double)((int)test[i] - (int)'0');
    i++;
  }
  for(int j=0;j<matrix->shape[0]*matrix->shape[1];j++){
    printf("%d",(int)matrix->matrix[j]);
  }
  printf("\n");
  return;
}

dARRAY * perform_dot(dARRAY * matrix){
  dARRAY * matrixT = transpose(matrix);
  dARRAY * result = dot(matrix,matrixT);
  free(matrixT);
  matrixT = NULL;
  return result;
}

int validate(dARRAY * matrix){
  int * matrix_int = (int*)malloc(sizeof(int)*1000);
  char * matrix_string = (char*)malloc(sizeof(char)*1000);
  int i=0;
  for(i=0; i<matrix->shape[0]*matrix->shape[1];i++){
    matrix_int[i] = (int)matrix->matrix[i];
  }
  int index = 0;
  char temp_char[1000];
  for(int j=0;j<matrix->shape[0]*matrix->shape[1];j++){
    long long int n = matrix_int[j];
    sprintf(temp_char,"%lld",n);
    for(int k=0;k<strlen(temp_char);k++){
      matrix_string[index++] = temp_char[k];
    }
  }
  free(matrix_int);
  matrix_int = NULL;
  // matrix_string[index++] = ' ';
  // matrix_string[index++] = '[';
  // sprintf(temp_char,"%d",matrix->shape[0]);
  // matrix_string[index++] = temp_char[0];
  // matrix_string[index++] = ',';
  // sprintf(temp_char,"%d",matrix->shape[1]);
  // matrix_string[index++] = temp_char[0];
  // matrix_string[index++] = ']';
  matrix_string[index] = '\n';
  shape(matrix);

  FILE * fp;
  char * line = NULL;
  size_t len = 0;
  ssize_t read;
  int comp_res = 0;
  fp = fopen("./src/test/_val_DOT.txt","r");
  while ((read = getline(&line, &len, fp)) != -1){
    comp_res = strcmp((const char*)matrix_string,(const char *)line);
    if(comp_res==0){ printf("valid\n"); break;}
  }
  fclose(fp);
  if (line)
      free(line);
  line = NULL;
  free(matrix_string);
  matrix_string = NULL;
  return !comp_res;
}

void write_DOT_TESTS(){
  FILE * file_ptr;
  file_ptr = fopen((const char * restrict)"./src/test/TEST_DOT.txt","a+");
  fputs("123456789 [3,3]\n",file_ptr);
  fputs("987654321 [3,3]\n",file_ptr);
  fputs("1000010000100001 [4,4]\n",file_ptr);
  fputs("5936493810481736 [4,4]\n",file_ptr);
  fputs("18427523783432189483 [5,4]\n",file_ptr);
  fputs("74198463784581376841849127382346864782368372846321 [10,5]\n",file_ptr);
  fputs("7419847419846378458137684184912738234686478236837284632163784581376841849127382346864782368372846321 [25,4]\n",file_ptr);
  fputs("463784581376841849127382346864782368372846321637844637845813768418491273823468647823683728463216378446378458137684184912738234686478236837284632163784 [25,6]\n",file_ptr);
  fputs("46378458137684184912738234686478236837284632163784 [50,1]\n",file_ptr);
  fputs("463784581374637845813768418491246378458137684184912738234686478236837284632163784738234686478236837284632163784684184912746378458137684184912738234686478236837284632163784382346864782368372846321637841947362718 [30,7]\n",file_ptr);
  fclose(file_ptr);
}

int main(){
  // TEST_DOT();
  int dims[] = {5,4};
  dARRAY * A = ones(dims);
  int dims2[] = {5,2};
  dARRAY * B = ones(dims2);
  dARRAY * res = add(A,B);
  free2d(A);
  free2d(B);
  free2d(res);
  return 0;
}