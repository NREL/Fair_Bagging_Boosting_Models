corrections: ./compiled_loss_funcs/corrections.cpp ./compiled_loss_funcs/vec.h ./compiled_loss_funcs/tools.h ./compiled_loss_funcs/mat.h
	g++ -c -fPIC ./compiled_loss_funcs/corrections.cpp -o ./compiled_loss_funcs/corrections.o
	g++ -shared -o ./compiled_loss_funcs/libcorrections.so ./compiled_loss_funcs/corrections.o

clean:
	rm ./compiled_loss_funcs/*.o
	rm ./compiled_loss_funcs/*.so