input_file = "requirements.txt"
output_file = "updated_reqvirenment.txt"

with open(input_file, 'r') as in_file:
    with open(output_file, 'w') as out_file:
        for line in in_file:
            # Remove version constraints by splitting at '==' and taking only the package name
            package_name = line.split('==')[0]
            out_file.write(package_name + '\n')

print(f"Updated requirements written to {output_file}")
