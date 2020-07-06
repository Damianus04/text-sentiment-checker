// 2. Type the keyword and push Enter key on keyboard
inputKeyword = document.getElementsByClassName("input_keyword");
inputKeyword.on("keyup", function (e) {
  if (e.keyCode === 13) {
    searchMovie();
  }
});
