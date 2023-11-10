const sliderInput = document.querySelector('.slider-input');
const sliderImage = document.querySelector('.slider-image');

sliderInput.addEventListener('input', () => {
  const imageIndex = sliderInput.value;
  const imageName = imageForIndex(imageIndex);
  sliderImage.setAttribute('src', imageName);
});

function imageForIndex(imageIndex) {
  const imageName = `../output/${imageIndex}.png`;
  return imageName;
}

document.addEventListener('DOMContentLoaded', () => {
  const imageName = imageForIndex(2);
  sliderImage.setAttribute('src', imageName);
});
