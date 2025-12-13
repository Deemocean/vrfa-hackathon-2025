import os
from dm_control import mjcf

def make_giant(xml_path, scale):
    print(f"Loading {xml_path}...")
    mjcf_model = mjcf.from_path(xml_path)
    
    # Scale all geoms
    print(f"Scaling geoms by {scale}...")
    for geom in mjcf_model.find_all('geom'):
        if geom.size is not None:
            geom.size = geom.size * scale
        if geom.pos is not None:
            geom.pos = geom.pos * scale
            
    # Scale all bodies (positions)
    print(f"Scaling body positions by {scale}...")
    for body in mjcf_model.find_all('body'):
        if body.pos is not None:
            body.pos = body.pos * scale
            
    # Note: Actuator gains/forces also need scaling for a realistic simulation!
    # This is left as an exercise for the hacker.
    
    return mjcf_model

if __name__ == "__main__":
    # Define paths
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    source_xml = os.path.join(base_dir, "assets", "humanoid", "humanoid.xml")
    output_dir = os.path.join(base_dir, "assets", "generated")
    output_xml_name = "humanoid_giant.xml"
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Run scaling
    giant_model = make_giant(source_xml, 2.0)
    
    # Save
    print(f"Saving to {os.path.join(output_dir, output_xml_name)}...")
    mjcf.export_with_assets(giant_model, out_dir=output_dir, out_file_name=output_xml_name)
    print("Done!")
