book_fname = 'data/Goblet.txt';
fid = fopen(book_fname,'r');
book_data = fscanf(fid, '%c');
fclose(fid);

book_chars = unique(book_data);

dimension = 81;

numbers = [1:dimension];

for i = 1 : dimension
    chars{i} = book_chars(i);
    numbers_cell{i} = numbers(i);
end
% char_to_ind = containers.Map('KeyType','char','ValueType','int32');
% ind_to_char = containers.Map('KeyType','int32','ValueType','char');
    

char_to_ind = containers.Map(chars,numbers);
ind_to_char = containers.Map(numbers,chars);

% keys(char_to_ind)
% values(char_to_ind)


% keys(ind_to_char)
% values(ind_to_char)