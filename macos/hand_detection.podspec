Pod::Spec.new do |s|
  s.name                  = 'hand_detection'
  s.version               = '2.2.0'
  s.summary               = 'Hand detection via TensorFlow Lite (macOS)'
  s.description           = 'Flutter plugin for on-device hand detection using TensorFlow Lite.'
  s.homepage              = 'https://github.com/your/repo'
  s.license               = { :type => 'MIT' }
  s.authors               = { 'You' => 'you@example.com' }
  s.source                = { :path => '.' }

  s.platform              = :osx, '11.0'

  # TFLite libraries are provided by flutter_litert dependency
end
